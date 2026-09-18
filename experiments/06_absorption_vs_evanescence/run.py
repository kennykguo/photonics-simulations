"""06. Absorption versus evanescence: two ways a field can decay, only one of them loses power.

Run with the Meep interpreter, from this directory:
    cd experiments/06_absorption_vs_evanescence && ../../.meep/bin/python run.py

Regenerates everything in out/ from scratch:
  * analytic numbers (notes 6, 25, 28): the doped ring's 125 dB/cm -> n'', alpha, a = e^{-alpha L/2}
  * 1-D Meep FDTD of a plane wave in a lossy medium n = n' - j n'' (decay along the flow)
  * 2-D Meep FDTD of a Si/SiO2 interface beyond the critical angle (decay across the flow)
  * time-averaged Poynting vectors computed from the E, H phasors of both runs
  * figures, an animation with a contact sheet, out/results.json, out/tools.json, out/results.txt
"""
import sys, pathlib, json, time, subprocess
sys.dont_write_bytecode = True                  # keep __pycache__ out of the experiment folder

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import meep as mp

from common import REF, use_style, SERIES, PALETTE
from common.units import db_per_cm_to_alpha_per_um
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()
matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
mp.verbosity(0)

from sims import run_lossy_plane_wave, run_tir_interface, LAM, OMEGA

T0 = time.time()
for stale in OUT.iterdir():                      # everything below is regenerated
    stale.unlink()

LOG = []
def say(*args):
    line = " ".join(str(a) for a in args)
    print(line); LOG.append(line)

def pct(sim, ref):
    return 100.0 * (sim - ref) / ref

# ----------------------------------------------------------------------------------------------
# 1. Analytic numbers (notes 6, 25, 28)
# ----------------------------------------------------------------------------------------------
say("== 1. Analytic numbers from the notes ==")
k0 = 2 * np.pi / LAM                                     # rad/µm
L_rt = REF.round_trip_um
alpha_doped = db_per_cm_to_alpha_per_um(REF.loss_db_cm_doped)     # 1/µm, intensity
alpha_pass = db_per_cm_to_alpha_per_um(REF.loss_db_cm_passive)
n_im_doped = alpha_doped * LAM / (4 * np.pi)              # notes 6: alpha = 4 pi n'' / lambda0
n_im_pass = alpha_pass * LAM / (4 * np.pi)
a_doped = np.exp(-alpha_doped * L_rt / 2)                 # notes 28: field retention per lap
a_pass = np.exp(-alpha_pass * L_rt / 2)
gamma_ref = k0 * np.sqrt(REF.neff ** 2 - REF.n_sio2 ** 2)     # notes 24: k_x = j gamma in the cladding
gamma_solver = k0 * np.sqrt(2.7 ** 2 - REF.n_sio2 ** 2)       # with the 08 solver value n_eff ~ 2.7
theta_c = np.degrees(np.arcsin(REF.n_sio2 / REF.n_si))
theta_ref = np.degrees(np.arcsin(REF.neff / REF.n_si))

say(f"doped ring  {REF.loss_db_cm_doped} dB/cm: alpha = {alpha_doped:.4e} 1/um = {alpha_doped*1e4:.2f} 1/cm, "
    f"n'' = {n_im_doped:.3e}, field decay length 2/alpha = {2/alpha_doped:.0f} um ({2/alpha_doped/L_rt:.1f} laps), "
    f"a = e^(-alpha L/2) = {a_doped:.4f} (REF {REF.a_round_trip}), power lost per lap 1-a^2 = {100*(1-a_doped**2):.1f} %")
say(f"passive     {REF.loss_db_cm_passive} dB/cm: alpha = {alpha_pass:.4e} 1/um, n'' = {n_im_pass:.3e}, a = {a_pass:.5f}, "
    f"lost per lap {100*(1-a_pass**2):.2f} %")
say(f"evanescent tail, n_eff {REF.neff}: gamma = {gamma_ref:.3f} 1/um, 1/gamma = {1e3/gamma_ref:.1f} nm; "
    f"with n_eff 2.7: 1/gamma = {1e3/gamma_solver:.1f} nm; critical angle {theta_c:.1f} deg, "
    f"n_eff {REF.neff} corresponds to theta = {theta_ref:.1f} deg")
say(f"length-scale contrast: evanescent 1/gamma = {1e3/gamma_ref:.0f} nm vs absorptive field decay 2/alpha = "
    f"{2/alpha_doped*1e3:.0f} nm -> ratio {2/alpha_doped*gamma_ref:.0f}")

# capstone: how much of the input power the doped ring absorbs at the operating point (all-pass ring,
# everything that does not reach the through port is absorbed inside the ring).  Upper bound on the
# self-heating: every absorbed milliwatt is assumed to heat the ring like a heater milliwatt does.
p_in_mw = 10 ** (REF.p_in_dbm / 10)
absorbed_frac = lambda delta_pm: (1 - REF.t_min) / (1 + (2 * delta_pm / REF.fwhm_pm) ** 2)   # 1 - T(delta), Lorentzian
f_res, f_opt = absorbed_frac(0.0), absorbed_frac(REF.delta_opt_pm)
p_abs_res_mw, p_abs_opt_mw = p_in_mw * f_res, p_in_mw * f_opt
dT_res, dT_opt = REF.r_th_K_per_mw * p_abs_res_mw, REF.r_th_K_per_mw * p_abs_opt_mw
say(f"capstone self-heating bound: P_in = {p_in_mw:.2f} mW; absorbed fraction on resonance {f_res:.3f} ({p_abs_res_mw:.2f} mW), at delta_opt "
    f"{REF.delta_opt_pm} pm {f_opt:.3f} ({p_abs_opt_mw:.2f} mW); if all of it heats the ring through R_th {REF.r_th_K_per_mw} K/mW: "
    f"dT = {dT_res:.1f} K (resonance) / {dT_opt:.1f} K (operating point) -> {dT_opt*REF.dlambda_dT_pm_per_K:.0f} pm red shift, "
    f"{dT_opt*REF.dlambda_dT_pm_per_K/REF.fwhm_pm:.2f} FWHM")

# ----------------------------------------------------------------------------------------------
# 2. Meep 1-D: plane wave in a lossy medium (decay ALONG the propagation)
# ----------------------------------------------------------------------------------------------
say("\n== 2. Meep 1-D: plane wave in n = n' - j n'' ==")
N_RE = REF.n_si
N_IM_VIS = 0.05                                          # visual case: strong loss so 10 um shows it
RES_1D = 100

def fit_decay(z, A, lo, hi):
    """Slope of ln A over lo < z < hi -> decay constant (positive = decaying)."""
    sel = (z > lo) & (z < hi)
    p = np.polyfit(z[sel], np.log(A[sel]), 1)
    return -p[0]

t = time.time()
vis = run_lossy_plane_wave(N_RE, N_IM_VIS, length_um=10.0, resolution=RES_1D)
zv = vis["z"]; zmax = zv.max()
alpha_vis_fit = fit_decay(zv, np.abs(vis["Ex"]) ** 2, 0.5, zmax - 0.3)
alpha_vis_an = 4 * np.pi * N_IM_VIS / LAM
kappa_vis_fit = fit_decay(zv, np.abs(vis["Ex"]), 0.5, zmax - 0.3)          # field decay = alpha/2
ph = np.unwrap(np.angle(vis["Ex"])); sel = (zv > 0.5) & (zv < zmax - 0.3)
beta_vis_fit = -np.polyfit(zv[sel], ph[sel], 1)[0]
divS = -np.gradient(vis["Sz"], zv)                                          # -dSz/dz = p_abs (notes 25)
ratio_divS = np.mean(divS[sel]) / np.mean(vis["p_abs"][sel])
say(f"visual case n = {N_RE} - j{N_IM_VIS}, {len(zv)} points over {zmax - zv.min():.1f} um of monitor "
    f"({zv.min():.2f} to {zmax:.2f} um past the source plane), {time.time()-t:.1f} s:")
say(f"  intensity alpha: FDTD {alpha_vis_fit:.4f} 1/um vs 4 pi n''/lambda = {alpha_vis_an:.4f} ({pct(alpha_vis_fit, alpha_vis_an):+.2f} %)")
say(f"  field decay:     FDTD {kappa_vis_fit:.4f} 1/um vs n'' k0 = {N_IM_VIS*k0:.4f} ({pct(kappa_vis_fit, N_IM_VIS*k0):+.2f} %) -> the factor 2 between field and power")
say(f"  phase constant:  FDTD {beta_vis_fit:.3f} rad/um vs n' k0 = {N_RE*k0:.3f} ({pct(beta_vis_fit, N_RE*k0):+.2f} %, FDTD numerical dispersion)")
say(f"  power balance:   <-dSz/dz> / <p_abs> = {ratio_divS:.4f} (expect 1: div S = -p_abs, energy leaves the field)")

# sweep n'' to show alpha is linear in n''
sweep_nim = np.array([0.01, 0.02, 0.05, 0.10])
sweep_alpha = []
for nim in sweep_nim:
    r = run_lossy_plane_wave(N_RE, nim, length_um=10.0, resolution=RES_1D)
    sweep_alpha.append(fit_decay(r["z"], np.abs(r["Ex"]) ** 2, 0.5, r["z"].max() - 0.3))
sweep_alpha = np.array(sweep_alpha)
sweep_an = 4 * np.pi * sweep_nim / LAM
say("  sweep n'' -> alpha (FDTD / analytic): " + ", ".join(f"{a:.4f}/{b:.4f}" for a, b in zip(sweep_alpha, sweep_an)))

# the real doped ring: n'' = 3.0e-4 over exactly one round trip (39.6 um) -> a straight from FDTD
t = time.time()
lap = run_lossy_plane_wave(N_RE, n_im_doped, length_um=L_rt + 3.0, resolution=RES_1D)   # monitor span = L_rt
zl = lap["z"] - lap["z"][0]
i0 = 0; i1 = np.argmin(np.abs(zl - L_rt))
a_fdtd = np.abs(lap["Ex"][i1]) / np.abs(lap["Ex"][i0])
alpha_lap_fit = fit_decay(zl, np.abs(lap["Ex"]) ** 2, 0.2, zl.max() - 0.2)
say(f"one round trip at the doped n'' = {n_im_doped:.2e} ({time.time()-t:.1f} s): |E({zl[i1]:.1f} um)| / |E(0)| = {a_fdtd:.4f} "
    f"vs a = {a_doped:.4f} ({pct(a_fdtd, a_doped):+.2f} %); fitted alpha {alpha_lap_fit*1e4:.2f} 1/cm = "
    f"{alpha_lap_fit*1e4*4.3429:.1f} dB/cm vs {REF.loss_db_cm_doped}")

# ----------------------------------------------------------------------------------------------
# 3. Meep 2-D: Si/SiO2 interface, plane wave beyond the critical angle (decay ACROSS the propagation)
# ----------------------------------------------------------------------------------------------
say("\n== 3. Meep 2-D: evanescent field at a Si/SiO2 interface ==")
RES_2D = 80
cases = {}
for label, neff in (("tir_ref", REF.neff), ("tir_solver", 2.7), ("subcritical", 1.0)):
    t = time.time()
    r = run_tir_interface(neff, resolution=RES_2D)
    x, z = r["x"], r["z"]
    clad = (x > 0.06) & (x < 0.9)
    A = np.abs(r["Ey"]).mean(axis=0)                                  # |E_y| vs x, averaged over z
    # Bloch consistency: k_point = beta forces the steady state to be f(x) e^{-j beta z}, so |E| must be
    # z-independent and the phase slope along z must be exactly -beta.  Neither is a physics result
    # (both are imposed by the boundary condition); they are a solver check, and the SIGN of the
    # recovered beta tests the conjugation bridge in sims.py (a +beta slope would mean e^{+j beta z}).
    zvar = (np.abs(r["Ey"]).std(axis=0) / np.abs(r["Ey"]).mean(axis=0)).max()
    r["z_variation"] = zvar
    rows = np.where(clad)[0][::8]                                     # a few x rows inside the cladding
    slopes = np.array([np.polyfit(z, np.unwrap(np.angle(r["Ey"][:, i])), 1)[0] for i in rows])
    r["beta_fit"] = float(-slopes.mean())                             # e^{-j beta z}: d(phase)/dz = -beta
    r["beta_fit_spread"] = float(slopes.std())
    r["beta_agreement_pct"] = pct(r["beta_fit"], r["beta"])
    if neff > REF.n_sio2:
        gam_fit = fit_decay(x, A, 0.06, 0.9)
        gam_an = k0 * np.sqrt(neff ** 2 - REF.n_sio2 ** 2)
        r["gamma_fit"], r["gamma_an"] = gam_fit, gam_an
        # standing wave in the silicon: incident + reflected plane waves with k_x = n1 k0 cos(theta) give nodes of
        # |E_y|(x) spaced pi/k_x = lambda0/(2 n1 cos theta).  Locate the dips of the z-averaged |E_y| for x < 0
        # (local minima, parabolic sub-grid refinement) and take the mean spacing.
        cos_th = np.sqrt(1 - (neff / REF.n_si) ** 2)
        r["node_spacing_an_nm"] = 1e3 * LAM / (2 * REF.n_si * cos_th)
        idx = [i for i in range(1, len(x) - 1) if x[i] < -0.05 and A[i] < A[i - 1] and A[i] < A[i + 1]]
        nodes = []
        for i in idx:
            y0, y1, y2 = A[i - 1], A[i], A[i + 1]
            dx = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)                          # vertex of the parabola through 3 points
            nodes.append(x[i] + dx * (x[1] - x[0]))
        r["nodes_x_um"] = nodes
        r["node_spacing_fdtd_nm"] = 1e3 * float(np.mean(np.diff(nodes))) if len(nodes) > 1 else None
    # Poynting checks in the cladding (notes 25)
    Sx_c, Sz_c = r["Sx"][:, clad], r["Sz"][:, clad]
    r["Sx_over_Sz_clad"] = Sx_c.mean() / Sz_c.mean()
    r["Sz_over_E2_clad"] = (Sz_c / np.abs(r["Ey"][:, clad]) ** 2).mean()
    r["Sz_over_E2_expected"] = r["beta"] / (2 * OMEGA)                # beta/(2 omega mu), mu = 1
    ratio_HzEy = (r["Hz"][:, clad] / r["Ey"][:, clad]).mean()
    r["HzEy_phase_deg"] = np.degrees(np.angle(ratio_HzEy))
    r["HzEy_mag"] = np.abs(ratio_HzEy)
    r["HxEy_real"] = float(np.real((r["Hx"][:, clad] / r["Ey"][:, clad]).mean()))   # notes 25: -beta/(omega mu)
    # divergence of <S>.  This is a CONSISTENCY row, not independent evidence: k_point makes every DFT quantity
    # z-independent, so dSz/dz = 0 by construction, and dSx/dx = 0 follows from <S_x> = 0 (TIR) or from S_x being
    # x-independent (uniform refracted wave, sub-critical).  It vanishes for both runs and does not discriminate.
    dSx = np.gradient(r["Sx"], x, axis=1); dSz = np.gradient(r["Sz"], z, axis=0)
    inner = (slice(2, -2), clad)
    r["divS_norm"] = np.abs((dSx + dSz)[inner]).max() / (np.abs(r["Sz"][inner]).max() * gamma_ref)
    theta = np.degrees(np.arcsin(neff / REF.n_si))
    say(f"{label}: n_eff {neff} (theta = {theta:.1f} deg, critical {theta_c:.1f}), grid {r['Ey'].shape}, {time.time()-t:.1f} s")
    say(f"  Bloch consistency (imposed by k_point, not physics): |E| varies along z by at most {zvar:.1e} (relative); "
        f"phase slope gives beta = {r['beta_fit']:.9f} rad/um vs imposed {r['beta']:.9f} ({r['beta_agreement_pct']:+.1e} %, "
        f"row spread {r['beta_fit_spread']:.1e}); the negative slope confirms e^(-j beta z), i.e. the conjugation bridge")
    if neff > REF.n_sio2:
        say(f"  decay length 1/gamma: FDTD {1e3/gam_fit:.1f} nm vs lambda/(2 pi sqrt(n_eff^2 - n2^2)) = {1e3/gam_an:.1f} nm ({pct(1/gam_fit, 1/gam_an):+.2f} %)")
        say(f"  standing-wave nodes in the silicon at x = " + ", ".join(f"{1e3*v:.0f}" for v in r["nodes_x_um"]) +
            f" nm: mean spacing {r['node_spacing_fdtd_nm']:.1f} nm vs lambda0/(2 n1 cos theta) = {r['node_spacing_an_nm']:.1f} nm "
            f"({pct(r['node_spacing_fdtd_nm'], r['node_spacing_an_nm']):+.2f} %)")
    say(f"  cladding <Sx>/<Sz> = {r['Sx_over_Sz_clad']:.2e}  (TIR: 0, no power crosses; sub-critical: k_x/k_z = "
        f"{np.sqrt(max(REF.n_sio2**2 - neff**2, 0))/neff:.3f})")
    say(f"  cladding <Sz>/|E_y|^2 = {r['Sz_over_E2_clad']:.4f} vs beta/(2 omega mu) = {r['Sz_over_E2_expected']:.4f}")
    say(f"  H_z / E_y in the cladding: phase {r['HzEy_phase_deg']:+.2f} deg, magnitude {r['HzEy_mag']:.3f} "
        f"(evanescent: -90 deg and gamma/omega = {np.sqrt(max(neff**2-REF.n_sio2**2,0))*k0/OMEGA:.3f})")
    say(f"  H_x / E_y in the cladding: {r['HxEy_real']:.4f} vs -beta/(omega mu) = {-r['beta']/OMEGA:.4f}")
    say(f"  max |div S| / (gamma max|Sz|) = {r['divS_norm']:.1e} (solver consistency, not independent physics: dSz/dz = 0 is "
        f"imposed by k_point and dSx/dx = 0 restates <Sx> = 0; the sub-critical run gives the same order)")
    cases[label] = r

tir = cases["tir_ref"]

# ----------------------------------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------------------------------
say("\n== 4. Figures ==")
BLUE, ORANGE, AQUA, VIOLET, RED, YELLOW = SERIES

# --- Figure 1: the two panels of the spec ------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 7.2), gridspec_kw=dict(height_ratios=[1.6, 1]))
ax = axs[0, 0]
Ex = vis["Ex"]; E0 = np.abs(Ex[0])
ax.plot(zv, np.real(Ex) / E0, color=BLUE, lw=1.2, label="Re{E_x(z)} at t = 0")
ax.plot(zv, np.abs(Ex) / E0, color=ORANGE, label="FDTD envelope |E_x(z)|")
ax.plot(zv, -np.abs(Ex) / E0, color=ORANGE)
ax.plot(zv, np.exp(-N_IM_VIS * k0 * zv) * np.exp(N_IM_VIS * k0 * zv[0]), "--", color=PALETTE["ink2"], lw=1, label="analytic e^(−n″k₀z)")
ax.set_xlabel("z along the propagation (µm)"); ax.set_ylabel("E_x / E_x(0)")
ax.set_title(f"(a) Absorption: n = {N_RE} − j{N_IM_VIS}, the wave decays ALONG its flow", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
ax = axs[1, 0]
ax.semilogy(zv, np.abs(Ex) ** 2 / E0 ** 2, color=BLUE, label="FDTD |E_x|²")
ax.semilogy(zv, np.exp(-alpha_vis_an * (zv - zv[0])), "--", color=PALETTE["ink2"], lw=1, label=f"e^(−αz), α = 4πn″/λ₀ = {alpha_vis_an:.3f} /µm")
ax.set_xlabel("z (µm)"); ax.set_ylabel("intensity (norm.)"); ax.legend(fontsize=8)
ax.set_title(f"Intensity falls as e^(−αz): FDTD α = {alpha_vis_fit:.3f} /µm ({pct(alpha_vis_fit, alpha_vis_an):+.1f} %)", fontsize=9)

ax = axs[0, 1]
x, z = tir["x"], tir["z"]
xs = (x > -0.6) & (x < 0.6)
m = np.abs(tir["Ey"][:, xs]).max()                       # colour scale: signed field / max|E_y|, centred on zero
F = np.real(tir["Ey"][:, xs]).T / m
im = ax.pcolormesh(z, x[xs], F, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
fig.colorbar(im, ax=ax, label="Re{E_y} / max|E_y|")
ax.axhline(0, color=PALETTE["ink"], lw=1.2)
ax.text(z.min() + 0.05, -0.5, "silicon n₁ = 3.50 (incident + reflected)", fontsize=8, color=PALETTE["ink"])
ax.text(z.min() + 0.05, 0.45, "silica n₂ = 1.45: evanescent e^(−γx)", fontsize=8, color=PALETTE["ink"])
ax.set_xlabel("z along the interface (µm)"); ax.set_ylabel("x normal to the interface (µm)")
ax.set_title(f"(b) Evanescence: TIR at θ = {theta_ref:.1f}° > θ_c = {theta_c:.1f}°, decay ACROSS the flow", fontsize=10)
ax = axs[1, 1]
A = np.abs(tir["Ey"]).mean(axis=0); A0 = A[np.argmin(np.abs(x))]
ax.semilogy(x * 1e3, A / A0, color=BLUE, label="FDTD |E_y|(x), z-averaged")
xc = x[x >= 0]
ax.semilogy(xc * 1e3, np.exp(-gamma_ref * xc), "--", color=PALETTE["ink2"], lw=1, label=f"e^(−γx), 1/γ = {1e3/gamma_ref:.0f} nm")
ax.axvline(0, color=PALETTE["ink"], lw=1)
ax.set_xlim(-600, 600); ax.set_ylim(1e-3, 2.5)
ax.set_xlabel("x (nm)"); ax.set_ylabel("|E_y| / |E_y(0)|"); ax.legend(fontsize=8, loc="upper right")
ax.set_title(f"|E| vs x: standing wave in Si, e^(−γx) in SiO₂; FDTD 1/γ = {1e3/tir['gamma_fit']:.0f} nm", fontsize=9)
fig.suptitle("Two decays that look alike on a plot and are physically opposite: absorption loses power, evanescence does not", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig1_absorption_vs_evanescence.png"); plt.close(fig)

# --- Figure 2: Poynting vectors -------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 7.6))
ax = axs[0, 0]
S0 = vis["Sz"][0]
ax.plot(zv, vis["Sz"] / S0, color=BLUE, label="⟨S_z⟩ = ½Re{E_x H_y*}")
ax.plot(zv, vis["p_abs"] / S0, color=ORANGE, label="p_abs = ½ω ε''|E|²")
ax.plot(zv, divS / S0, ":", color=VIOLET, lw=2.5, label="−d⟨S_z⟩/dz  (= −∇·S)")
ax.set_xlabel("z (µm)"); ax.set_ylabel("normalised to ⟨S_z⟩(0)")
ax.set_title(f"Absorber: power flow shrinks and ∇·S = −p_abs < 0 (ratio {ratio_divS:.3f})", fontsize=10)
ax.legend(fontsize=8)

ax = axs[0, 1]
prof_Sz = tir["Sz"].mean(axis=0); prof_Sx = tir["Sx"].mean(axis=0)
Sn = np.abs(prof_Sz).max()
ax.plot(x * 1e3, prof_Sz / Sn, color=BLUE, label="⟨S_z⟩ along the interface")
ax.plot(x * 1e3, prof_Sx / Sn, color=ORANGE, label="⟨S_x⟩ across the interface")
ax.axvline(0, color=PALETTE["ink"], lw=1); ax.set_xlim(-600, 600)
ax.set_xlabel("x (nm)"); ax.set_ylabel("⟨S⟩ / max ⟨S_z⟩")
ax.set_title(f"Evanescent: ⟨S_x⟩ = 0 everywhere, ⟨S_z⟩ > 0 in the tail (⟨S_x⟩/⟨S_z⟩ = {tir['Sx_over_Sz_clad']:.0e})", fontsize=10)
ax.legend(fontsize=8)

def quiver_panel(fig, ax, r, title):
    x, z = r["x"], r["z"]
    xs = (x > -0.7) & (x < 0.7)
    mag = np.hypot(r["Sx"], r["Sz"])[:, xs].T
    im = ax.pcolormesh(z, x[xs], mag / mag.max(), cmap="Blues", vmin=0, vmax=1, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, label="|⟨S⟩| / max|⟨S⟩|")
    st = 12
    Z, X = np.meshgrid(z[::st], x[xs][::st])
    U = r["Sz"][:, xs][::st, ::st].T; V = r["Sx"][:, xs][::st, ::st].T
    Mg = np.hypot(U, V); keep = Mg > 1e-2 * mag.max()            # stop where the tail has visibly ended
    scale = np.sqrt(Mg / mag.max())                                # arrow length ∝ sqrt|S| (unit direction × sqrt magnitude)
    ax.quiver(Z[keep], X[keep], (U / Mg * scale)[keep], (V / Mg * scale)[keep], color=PALETTE["ink"], scale=22,
              width=0.004, headwidth=4)
    ax.axhline(0, color=PALETTE["ink"], lw=1.2)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("x (µm)"); ax.set_title(title, fontsize=10)

quiver_panel(fig, axs[1, 0], tir, f"⟨S⟩ arrows (length ∝ √|⟨S⟩|), θ = {theta_ref:.1f}° > θ_c: parallel in silica")
sub = cases["subcritical"]
quiver_panel(fig, axs[1, 1], sub, f"Same, θ = {np.degrees(np.arcsin(1.0/REF.n_si)):.1f}° < θ_c: power crosses into silica (S_x/S_z = {sub['Sx_over_Sz_clad']:.2f})")
fig.suptitle("Time-averaged Poynting vector ⟨S⟩ = ½Re{E × H*} from the FDTD phasors (notes 25)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig2_poynting.png"); plt.close(fig)

# --- Figure 3: alpha vs n'' and the one-lap doped-ring run -----------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axs[0]
nn = np.linspace(0, 0.11, 50)
ax.plot(nn, 4 * np.pi * nn / LAM, "--", color=PALETTE["ink2"], lw=1, label="α = 4πn″/λ₀ (notes 6)")
ax.plot(sweep_nim, sweep_alpha, "o", color=BLUE, ms=7, label="Meep 1-D fit")
ax.set_xlabel("n″ (imaginary part of the index)"); ax.set_ylabel("intensity decay α (1/µm)")
ax.set_title("α is linear in n″: FDTD points sit on the notes' line", fontsize=10); ax.legend(fontsize=8)
ax = axs[1]
E_lap = np.abs(lap["Ex"]) / np.abs(lap["Ex"][0]); E_an = np.exp(-alpha_doped * zl / 2)
ax.plot(zl, E_lap, color=BLUE, lw=2.8, label=f"FDTD |E| at n″ = {n_im_doped:.1e} (125 dB/cm)")
ax.plot(zl, E_an, "--", color=VIOLET, lw=1.2, label="analytic e^(−αz/2), notes 28 (dashed, on top)")
ax.axvline(L_rt, color=ORANGE, lw=1, label=f"L = {L_rt} µm, a = {a_doped:.4f}"); ax.axhline(a_doped, color=ORANGE, lw=1)
ax.annotate(f"one round trip L = {L_rt} µm\na = |E(L)|/|E(0)| = {a_fdtd:.4f}\n(notes 28: {a_doped:.4f})",
            xy=(L_rt, a_doped), xytext=(26, 0.966), fontsize=8, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
ax.set_xlabel("distance travelled z (µm)"); ax.set_ylabel("|E| / |E(0)|"); ax.set_ylim(0.935, 1.003)
ax.set_title("The doped ring's field after one lap: 5.5 % of the amplitude is gone", fontsize=10); ax.legend(fontsize=8, loc="upper right")
ins = ax.inset_axes([0.10, 0.19, 0.36, 0.27])                    # residual: the two curves are indistinguishable at full scale
ins.plot(zl, 100 * (E_lap / E_an - 1), color=BLUE, lw=1)
ins.axhline(0, color=VIOLET, lw=0.8, ls="--"); ins.axvline(L_rt, color=ORANGE, lw=0.8)
ins.set_title(f"FDTD / analytic − 1 (%): {pct(a_fdtd, a_doped):+.3f} % at L", fontsize=7, pad=2)
ins.set_xlabel("z (µm)", fontsize=7, labelpad=1); ins.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig3_alpha_and_one_lap.png"); plt.close(fig)

# --- Figure 4: the two length scales and the ring loss budget --------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axs[0]
d = np.logspace(-2, 3.5, 400)   # µm
ax.semilogx(d, np.exp(-gamma_ref * d), color=ORANGE, label=f"evanescent |E| = e^(−γx), 1/γ = {1e3/gamma_ref:.0f} nm (lossless)")
ax.semilogx(d, np.exp(-alpha_doped * d / 2), color=BLUE, label=f"doped ring |E| = e^(−αz/2), 2/α = {2/alpha_doped:.0f} µm (heat)")
ax.semilogx(d, np.exp(-alpha_pass * d / 2), color=AQUA, label=f"passive strip, 2/α = {2/alpha_pass*1e-4:.1f} cm")
for k in (1, 10, 100):
    ax.axvline(k * L_rt, color=PALETTE["line"], lw=1)
    ax.text(k * L_rt * 1.06, 0.98, f"{k} lap{'s' if k>1 else ''}", rotation=90, fontsize=7, color=PALETTE["ink2"], va="top", ha="left")
ax.set_xlabel("distance (µm): x across the interface, or z along the ring"); ax.set_ylabel("field amplitude (norm.)")
ax.set_title(f"Same exponential, {2/alpha_doped*gamma_ref:.0f}× different length scale, opposite physics", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")
ax = axs[1]
db = np.logspace(0, 3, 300)
a_of = np.exp(-db_per_cm_to_alpha_per_um(db) * L_rt / 2)
ax.semilogx(db, 100 * (1 - a_of ** 2), color=BLUE, label="power absorbed per lap, 1 − a²")
ax.semilogx(db, 100 * (1 - a_of), color=ORANGE, label="field lost per lap, 1 − a")
for val, lab, c in ((REF.loss_db_cm_doped, f"doped {REF.loss_db_cm_doped:.0f} dB/cm: a = {a_doped:.3f}", BLUE),
                    (REF.loss_db_cm_passive, f"passive {REF.loss_db_cm_passive:.0f} dB/cm: a = {a_pass:.4f}", AQUA)):
    ax.axvline(val, color=c, lw=1, ls=":"); ax.text(val * 1.08, 40, lab, fontsize=8, rotation=90, color=c, va="center")
ax.set_xlabel("propagation loss (dB/cm)"); ax.set_ylabel("lost per round trip of 39.6 µm (%)")
ax.set_title("Notes 28: a = e^(−αL/2) turns dB/cm into a per-lap retention", fontsize=10); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig4_length_scales_and_ring_budget.png"); plt.close(fig)
say("figures written")

# ----------------------------------------------------------------------------------------------
# 5. Animation: the physical fields Re{E e^{jωt}} side by side, plus a contact sheet
# ----------------------------------------------------------------------------------------------
say("\n== 5. Animation ==")
N_FRAMES, FPS, N_PERIODS = 180, 30, 3
phases = 2 * np.pi * N_PERIODS * np.arange(N_FRAMES) / N_FRAMES
T_fs = LAM * 1e-6 / 299_792_458.0 * 1e15                 # optical period in fs

def make_frame_fig():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a1.plot(zv, np.abs(Ex) / E0, color=ORANGE, lw=1.2); a1.plot(zv, -np.abs(Ex) / E0, color=ORANGE, lw=1.2)
    (ln,) = a1.plot(zv, np.real(Ex) / E0, color=BLUE, lw=1.4)
    a1.set_ylim(-1.1, 1.1); a1.set_xlabel("z along the flow (µm)"); a1.set_ylabel("E_x / E_x(0)")
    a1.set_title(f"(a) absorber n = {N_RE} − j{N_IM_VIS}: crests move +z inside a FIXED shrinking envelope", fontsize=9.5)
    qm = a2.pcolormesh(z, x[xs], F, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
    fig.colorbar(qm, ax=a2, label="E_y / max|E_y|")            # fixed vmin/vmax so the scale is static
    a2.axhline(0, color=PALETTE["ink"], lw=1.2)
    a2.set_xlabel("z along the interface (µm)"); a2.set_ylabel("x (µm)")
    a2.set_title(f"(b) TIR θ = {theta_ref:.0f}°: crests slide along z, the e^(−γx) tail never moves out", fontsize=9.5)
    a2.text(z.min() + 0.05, -0.55, "Si", fontsize=9); a2.text(z.min() + 0.05, 0.5, "SiO₂", fontsize=9)
    txt = fig.text(0.5, 0.965, "", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, ln, qm, txt

def draw(fig_objs, ph):
    fig, ln, qm, txt = fig_objs
    ln.set_ydata(np.real(Ex * np.exp(1j * ph)) / E0)
    qm.set_array((np.real(tir["Ey"][:, xs] * np.exp(1j * ph)).T / m).ravel())
    txt.set_text(f"Physical field Re{{Ẽ e^{{jωt}}}} at t = {ph/(2*np.pi)*T_fs:.2f} fs  (period T = {T_fs:.2f} fs at 1310 nm)")
    return ln, qm, txt

objs = make_frame_fig()
anim = FuncAnimation(objs[0], lambda i: draw(objs, phases[i]), frames=N_FRAMES, blit=False)
anim.save(OUT / "absorption_vs_evanescence.mp4", writer=FFMpegWriter(fps=FPS, bitrate=2500))
plt.close(objs[0])

# contact sheet: 6 stills over one optical period, each still = (absorber line, TIR map)
fig, axs = plt.subplots(3, 4, figsize=(14, 9))
for k, ph in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False)):
    ax = axs[k // 2, 2 * (k % 2)]
    ax.plot(zv, np.abs(Ex) / E0, color=ORANGE, lw=1); ax.plot(zv, -np.abs(Ex) / E0, color=ORANGE, lw=1)
    ax.plot(zv, np.real(Ex * np.exp(1j * ph)) / E0, color=BLUE, lw=1.2)
    ax.set_ylim(-1.1, 1.1); ax.set_xlim(0, 4)
    ax.set_title(f"t = {ph/(2*np.pi)*T_fs:.2f} fs = {ph/(2*np.pi):.2f} T: absorber", fontsize=9)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("E / E(0)")
    ax = axs[k // 2, 2 * (k % 2) + 1]
    im = ax.pcolormesh(z, x[xs], np.real(tir["Ey"][:, xs] * np.exp(1j * ph)).T / m, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, label="E_y / max|E_y|")
    ax.axhline(0, color=PALETTE["ink"], lw=1); ax.grid(False)
    ax.set_title(f"t = {ph/(2*np.pi):.2f} T: TIR interface", fontsize=9)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("x (µm)")
fig.suptitle("Stills from absorption_vs_evanescence.mp4: the absorber's crests advance inside a fixed envelope; the TIR crests slide along z with a fixed profile in x", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "absorption_vs_evanescence_frames.png"); plt.close(fig)
dur = subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                      str(OUT / "absorption_vs_evanescence.mp4")], capture_output=True, text=True).stdout.strip()
say(f"video written: {N_FRAMES} frames at {FPS} fps, duration {dur} s")

# ----------------------------------------------------------------------------------------------
# 6. results.json, tools.json, results.txt
# ----------------------------------------------------------------------------------------------
results = {
    "lambda_nm": REF.lambda_nm,
    "analytic": {
        "alpha_doped_per_cm": alpha_doped * 1e4, "alpha_doped_per_um": alpha_doped,
        "n_imag_doped": n_im_doped, "a_doped": a_doped, "a_doped_ref": REF.a_round_trip,
        "a_doped_agreement_pct": pct(a_doped, REF.a_round_trip),
        "power_lost_per_lap_doped_pct": 100 * (1 - a_doped ** 2),
        "field_decay_length_doped_um": 2 / alpha_doped, "field_decay_length_doped_laps": 2 / alpha_doped / L_rt,
        "alpha_passive_per_cm": alpha_pass * 1e4, "n_imag_passive": n_im_pass, "a_passive": a_pass,
        "power_lost_per_lap_passive_pct": 100 * (1 - a_pass ** 2),
        "gamma_ref_per_um": gamma_ref, "decay_length_ref_nm": 1e3 / gamma_ref,
        "decay_length_neff_2p7_nm": 1e3 / gamma_solver,
        "critical_angle_deg": theta_c, "theta_for_neff_ref_deg": theta_ref,
        "si_node_spacing_nm": 1e3 * LAM / (2 * REF.n_si * np.cos(np.radians(theta_ref))),
        "length_scale_ratio_absorptive_over_evanescent": 2 / alpha_doped * gamma_ref,
        "self_heating_bound": {"p_in_mw": p_in_mw, "absorbed_fraction_on_resonance": f_res,
                               "absorbed_fraction_at_delta_opt": f_opt, "absorbed_mw_on_resonance": p_abs_res_mw,
                               "absorbed_mw_at_delta_opt": p_abs_opt_mw, "dT_K_on_resonance": dT_res,
                               "dT_K_at_delta_opt": dT_opt, "shift_pm_at_delta_opt": dT_opt * REF.dlambda_dT_pm_per_K,
                               "assumption": "all-pass ring; all non-transmitted power absorbed in the ring; every absorbed mW heats like a heater mW (R_th)"},
    },
    "meep_1d_absorber": {
        "n_re": N_RE, "n_im_visual": N_IM_VIS, "resolution_px_per_um": RES_1D,
        "alpha_fdtd_per_um": alpha_vis_fit, "alpha_analytic_per_um": alpha_vis_an, "alpha_agreement_pct": pct(alpha_vis_fit, alpha_vis_an),
        "field_decay_fdtd_per_um": kappa_vis_fit, "field_decay_analytic_per_um": N_IM_VIS * k0,
        "beta_fdtd_per_um": beta_vis_fit, "beta_analytic_per_um": N_RE * k0, "beta_agreement_pct": pct(beta_vis_fit, N_RE * k0),
        "divS_over_pabs": ratio_divS,
        "sweep_n_im": sweep_nim.tolist(), "sweep_alpha_fdtd": sweep_alpha.tolist(), "sweep_alpha_analytic": sweep_an.tolist(),
        "one_lap": {"n_im": n_im_doped, "span_um": float(zl[i1]), "a_fdtd": a_fdtd, "a_analytic": a_doped,
                    "agreement_pct": pct(a_fdtd, a_doped), "alpha_fit_db_per_cm": alpha_lap_fit * 1e4 * 4.3429},
    },
    "meep_2d_interface": {label: {
        "neff": r["neff"], "theta_deg": float(np.degrees(np.arcsin(r["neff"] / REF.n_si))), "resolution_px_per_um": RES_2D,
        "bloch_check_relative_z_variation_of_E": float(r["z_variation"]),
        "bloch_check_beta_fdtd_per_um": r["beta_fit"], "beta_imposed_per_um": float(r["beta"]),
        "bloch_check_beta_agreement_pct": float(r["beta_agreement_pct"]), "bloch_check_beta_row_spread_per_um": r["beta_fit_spread"],
        "decay_length_fdtd_nm": (1e3 / r["gamma_fit"]) if "gamma_fit" in r else None,
        "decay_length_analytic_nm": (1e3 / r["gamma_an"]) if "gamma_an" in r else None,
        "decay_length_agreement_pct": (pct(1 / r["gamma_fit"], 1 / r["gamma_an"])) if "gamma_fit" in r else None,
        "Sx_over_Sz_cladding": float(r["Sx_over_Sz_clad"]),
        "Sz_over_E2_cladding": float(r["Sz_over_E2_clad"]), "Sz_over_E2_expected": float(r["Sz_over_E2_expected"]),
        "Hz_over_Ey_phase_deg": float(r["HzEy_phase_deg"]), "Hz_over_Ey_mag": float(r["HzEy_mag"]),
        "Hx_over_Ey": r["HxEy_real"], "Hx_over_Ey_expected": float(-r["beta"] / OMEGA),
        "node_spacing_fdtd_nm": r.get("node_spacing_fdtd_nm"), "node_spacing_analytic_nm": r.get("node_spacing_an_nm"),
        "node_spacing_agreement_pct": (pct(r["node_spacing_fdtd_nm"], r["node_spacing_an_nm"]) if r.get("node_spacing_fdtd_nm") else None),
        "nodes_x_um": r.get("nodes_x_um"),
        "consistency_check_divS_normalised": float(r["divS_norm"]), "meep_time": float(r["meep_time"]),
    } for label, r in cases.items()},
    "runtime_s": time.time() - T0,
}
(OUT / "results.json").write_text(json.dumps(results, indent=2))

RUNTIME_PHRASE = "about 100 s on this laptop, 75 to 130 s depending on load"   # quoted verbatim in the README
tools = [
    {"tool": "Meep (pymeep, FDTD)", "version": mp.__version__,
     "what_it_is": "Meep is MIT's open-source finite-difference time-domain solver: it steps Maxwell's curl equations on a Yee grid in real time and is the standard free tool for photonic-device simulation (waveguides, resonators, scattering, absorption).",
     "used_for": "Two tiny problems. (1) A 1-D cell filled with n = n' - j n'' (loss entered as Meep's D_conductivity, sigma_D = omega*eps''/eps_inf), a Gaussian pulse and a DFT monitor giving the steady-state phasors E_x(z), H_y(z), for a visual case n'' = 0.05, a sweep n'' = 0.01..0.10, and the doped ring's own n'' = 3.0e-4 over exactly one round trip of 39.6 um. (2) A 2-D cell, Bloch-periodic along the interface with k_z = n_eff k0, a Si/SiO2 boundary, a plane-wave line source in the silicon; run beyond the critical angle (n_eff 2.5 and 2.7) and below it (n_eff 1.0). The E and H phasors give <S> = 1/2 Re{E x H*}.",
     "result": f"Absorber: alpha = {alpha_vis_fit:.4f} /um vs 4 pi n''/lambda = {alpha_vis_an:.4f} ({pct(alpha_vis_fit, alpha_vis_an):+.2f} %); -dSz/dz / p_abs = {ratio_divS:.3f}; one lap at 125 dB/cm: a = {a_fdtd:.4f} vs {a_doped:.4f} ({pct(a_fdtd, a_doped):+.2f} %). Interface: 1/gamma = {1e3/tir['gamma_fit']:.1f} nm vs {1e3/tir['gamma_an']:.1f} nm ({pct(1/tir['gamma_fit'], 1/tir['gamma_an']):+.2f} %); cladding <Sx>/<Sz> = {tir['Sx_over_Sz_clad']:.0e}; H_z/E_y phase {tir['HzEy_phase_deg']:+.1f} deg; Bloch consistency (imposed, not physics): |E| varies along z by {tir['z_variation']:.0e}, phase slope beta = {tir['beta_fit']:.6f} vs imposed {tir['beta']:.6f} rad/um ({tir['beta_agreement_pct']:+.0e} %).",
     "how_to_observe": f"cd experiments/06_absorption_vs_evanescence && ../../.meep/bin/python run.py ({RUNTIME_PHRASE}). Look at out/fig1_absorption_vs_evanescence.png (the two decays), out/fig2_poynting.png (S arrows and profiles), out/fig3_alpha_and_one_lap.png, out/absorption_vs_evanescence.mp4. Change N_IM_VIS, the n_eff list in section 3 of run.py, or RES_1D/RES_2D and re-run; the fitted numbers are reprinted and rewritten to out/results.json."},
    {"tool": "NumPy (analytic layer and fits)", "version": np.__version__,
     "what_it_is": "NumPy is the array library every Python simulation is built on; here it is also the 'pen and paper' half of the experiment (the notes' formulas evaluated numerically) and does the fits of the FDTD data (polyfit of ln|E|).",
     "used_for": "alpha = loss_dB/cm / 4.343, n'' = alpha*lambda/(4 pi), a = e^{-alpha L/2}, gamma = k0 sqrt(n_eff^2 - n2^2), the self-heating bound, and least-squares slopes of ln|E|^2 vs z and ln|E| vs x, the unwrapped phase slope for beta (1-D absorber and 2-D Bloch check), the dips of |E_y|(x) in the silicon (standing-wave node spacing, parabolic sub-grid refinement), the numerical divergence of <S> (consistency row only).",
     "result": f"125 dB/cm -> alpha = {alpha_doped*1e4:.2f} /cm, n'' = {n_im_doped:.2e}, a = {a_doped:.4f} (REF {REF.a_round_trip}, {pct(a_doped, REF.a_round_trip):+.2f} %); 1/gamma = {1e3/gamma_ref:.1f} nm for n_eff 2.5 ({1e3/gamma_solver:.1f} nm for 2.7); the absorptive field length 2/alpha = {2/alpha_doped:.0f} um is {2/alpha_doped*gamma_ref:.0f}x the evanescent one; Si node spacing lambda0/(2 n1 cos theta) = {1e3 * LAM / (2 * REF.n_si * np.cos(np.radians(theta_ref))):.1f} nm (FDTD {tir['node_spacing_fdtd_nm']:.1f} nm); absorbed at delta_opt {p_abs_opt_mw:.3f} mW.",
     "how_to_observe": "Section 1 of run.py prints them (also in out/results.txt and out/results.json). Edit REF.loss_db_cm_doped indirectly by passing a different dB/cm to db_per_cm_to_alpha_per_um, or change REF.neff in the gamma line, and re-run."},
    {"tool": "matplotlib (figures + FuncAnimation/FFMpegWriter)", "version": matplotlib.__version__,
     "what_it_is": "The standard Python plotting library; its animation module writes frame sequences to video through ffmpeg.",
     "used_for": "Four figures (RdBu_r for the signed field maps with a colorbar centred on zero, Blues for |S|, quiver for the Poynting arrows with length proportional to sqrt|S|) and a 6 s two-panel animation of the physical field Re{E e^{j omega t}} for the absorber and for the TIR interface, plus a contact sheet of six stills.",
     "result": "out/fig1..fig4 png, out/absorption_vs_evanescence.mp4 (6 s, 30 fps, three optical periods of 4.37 fs), out/absorption_vs_evanescence_frames.png.",
     "how_to_observe": "open out/absorption_vs_evanescence.mp4: on the left the crests move to the right inside an envelope that does not move; on the right the crests slide along the interface while the amplitude profile across it stays frozen. Change N_PERIODS or N_FRAMES in section 5 of run.py."},
    {"tool": "ffmpeg / ffprobe", "version": subprocess.run(["/opt/homebrew/bin/ffmpeg", "-version"], capture_output=True, text=True).stdout.split()[2],
     "what_it_is": "The universal command-line video encoder; matplotlib pipes rendered frames into it, ffprobe reads the metadata back.",
     "used_for": "Encoding the mp4 (H.264) and checking its duration.",
     "result": f"absorption_vs_evanescence.mp4, {dur} s.",
     "how_to_observe": "/opt/homebrew/bin/ffprobe out/absorption_vs_evanescence.mp4"},
]
(OUT / "tools.json").write_text(json.dumps(tools, indent=2))
say(f"\nTotal runtime {time.time()-T0:.0f} s")
(OUT / "results.txt").write_text("\n".join(LOG) + "\n")
