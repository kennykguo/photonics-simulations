"""Experiment 11: the evanescent directional coupler (notes §26, §27).

Run with the Meep interpreter from this directory:
    cd experiments/11_directional_coupler && ../../.meep/bin/python run.py

Stages (all outputs regenerated in out/):
  A. analytic: single 220 nm TE slab, even/odd supermodes of two slabs (5-layer transfer matrix),
     κ_c = (β+ − β−)/2, CMT closed form, κ_c versus gap  (slab_coupler.py, numpy/scipy)
  B. Meep eigenmode solver: β± of the coupled slabs on the FDTD grid, mode profiles
  C. Meep 2-D FDTD, continuous wave at 1310 nm, gaps 150 / 200 / 300 nm: |E| maps, modal power in
     each guide versus z from the time-averaged Poynting flux, fit to cos²/sin²(κ_c z)
  D. Meep 2-D FDTD pulse: video of a pulse hopping between the guides (+ contact sheet)
  E. CW phase video: Re{Ẽ e^{jωt}} in a 6 µm window, phase fronts move, envelope does not
  F. SAX (sax_part.py, .venv): S-matrix, unitarity, interference, critical-coupling lottery
  G. gdsfactory (gds_part.py, .venv): ring-bus point coupler + straight coupler GDS/PNG, L_eff
  H. results.json, tools.json, results.txt
"""
import sys, pathlib, json, time, subprocess, shutil
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.optimize import curve_fit
from scipy.signal import hilbert
import meep as mp
from common.params import k0_per_um
import slab_coupler as sc
import fdtd_coupler as fc

HERE = pathlib.Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
T0 = time.time()
LAM = REF.lambda_nm * 1e-3              # 1.31 µm
K0 = k0_per_um(LAM * 1e3)               # 4.796 rad/µm; derived from LAM so every stage follows the same wavelength
RES = 40                                # FDTD pixels per µm (220 nm slab = 8.8 px)
COURANT = 0.5                           # Meep's default Courant factor S = cΔt/Δx (used only for the dispersion estimate)
fc.set_wavelength(LAM)                  # the FDTD module follows this file's wavelength too
GAPS_FDTD = [0.15, 0.20, 0.30]
GAPS_EIG = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
timing = {}
results = {"inputs": dict(lambda_nm=REF.lambda_nm, n_si=REF.n_si, n_sio2=REF.n_sio2, slab_thickness_nm=220.0,
                          fdtd_resolution_px_per_um=RES, fdtd_cell_um=[fc.SX, fc.SY], pml_um=fc.PML,
                          gaps_fdtd_nm=[g * 1e3 for g in GAPS_FDTD], gaps_eigen_nm=[g * 1e3 for g in GAPS_EIG],
                          convention="e^{jωt}, +z wave e^{-jβz}; Meep phasors are conjugated to this convention")}

# ------------------------------------------------------------------ clean out/ (regenerate from scratch)
for p in OUT.iterdir():
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


# ================================================================== A. analytic
t = time.time()
slab = sc.single_slab_te(REF.n_si, REF.n_sio2, 0.22, LAM)
results["single_slab_analytic"] = {k: float(v) for k, v in slab.items()}
results["single_slab_analytic"]["decay_length_nm"] = float(1e3 / slab["gamma"])
gaps_fine = np.linspace(0.05, 0.50, 91)
sm_fine = [sc.supermodes(g, REF.n_si, REF.n_sio2, 0.22, LAM) for g in gaps_fine]
kappa_tm = np.array([s["kappa"] for s in sm_fine])
kappa_cmt = np.array([sc.cmt_kappa(g, REF.n_si, REF.n_sio2, 0.22, LAM) for g in gaps_fine])
sel = (gaps_fine >= 0.15) & (gaps_fine <= 0.40)
slope, icpt = np.polyfit(gaps_fine[sel], np.log(kappa_tm[sel]), 1)
results["kappa_vs_gap_analytic"] = dict(
    gaps_um=gaps_fine.tolist(), kappa_supermode_per_um=kappa_tm.tolist(), kappa_cmt_per_um=kappa_cmt.tolist(),
    gamma_fit_from_log_slope_per_um=float(-slope), gamma_single_slab_per_um=float(slab["gamma"]),
    gamma_fit_agreement_pct=float(100 * (-slope) / slab["gamma"]),
    cmt_vs_supermode_max_rel_dev_pct=float(100 * np.max(np.abs(kappa_cmt / kappa_tm - 1))),
    per_gap={f"{g*1e3:.0f}nm": dict(n_even=float(s["n_even"]), n_odd=float(s["n_odd"]), kappa_per_um=float(s["kappa"]),
                                    L_c_um=float(s["L_c"]), kappa_cmt_per_um=float(sc.cmt_kappa(g, REF.n_si, REF.n_sio2, 0.22, LAM)))
             for g, s in ((g, sc.supermodes(g, REF.n_si, REF.n_sio2, 0.22, LAM)) for g in sorted(set(GAPS_FDTD + GAPS_EIG)))})
timing["A_analytic_s"] = time.time() - t
log(f"A analytic: n_eff = {slab['n_eff']:.4f}, γ = {slab['gamma']:.3f}/µm (1/γ = {1e3/slab['gamma']:.1f} nm); "
    f"ln κ slope gives γ = {-slope:.3f}/µm; CMT vs transfer matrix within {results['kappa_vs_gap_analytic']['cmt_vs_supermode_max_rel_dev_pct']:.2f} %")

# ================================================================== B. Meep eigenmode supermodes
t = time.time()
n_single_meep = fc.eigen_single(RES)
eig = {}
for g in GAPS_EIG:
    e = fc.eigen_supermodes(g, RES)
    kap = K0 * (e["even"]["n_eff"] - e["odd"]["n_eff"]) / 2
    an = sc.supermodes(g, REF.n_si, REF.n_sio2, 0.22, LAM)
    eig[f"{g*1e3:.0f}nm"] = dict(n_even=float(e["even"]["n_eff"]), n_odd=float(e["odd"]["n_eff"]), kappa_per_um=float(kap),
                                L_c_um=float(np.pi / (2 * kap)), kappa_vs_analytic_pct=float(100 * kap / an["kappa"]),
                                n_even_minus_analytic=float(e["even"]["n_eff"] - an["n_even"]))
    if abs(g - 0.20) < 1e-9:
        eig_prof = e
results["meep_eigenmode"] = dict(single_slab_n_eff=float(n_single_meep),
                                 single_slab_vs_analytic_pct=float(100 * n_single_meep / slab["n_eff"]), per_gap=eig)
timing["B_eigenmode_s"] = time.time() - t
log(f"B Meep eigenmode: single slab n_eff {n_single_meep:.4f} (analytic {slab['n_eff']:.4f}); "
    f"gap 200 nm κ = {eig['200nm']['kappa_per_um']:.4f}/µm ({eig['200nm']['kappa_vs_analytic_pct']:.1f} % of analytic)")

# ================================================================== C. FDTD continuous wave
def sin2(z, A, kap, z0, c):
    return A * np.sin(kap * (z - z0)) ** 2 + c


cw = {}
fdtd = {}
t = time.time()
for g in GAPS_FDTD:
    r = fc.run_cw(g, RES)
    x, y = r["x"], r["y"]
    z = x - fc.X_START                          # z = 0 where guide 2 begins
    P0 = r["P1"] + r["P2"]
    win = (z > 0.5) & (x < fc.SX / 2 - fc.PML - 0.5)
    frac2 = r["P2"] / P0
    an = sc.supermodes(g, REF.n_si, REF.n_sio2, 0.22, LAM)
    popt, _ = curve_fit(sin2, z[win], frac2[win], p0=[1.0, an["kappa"], 0.0, 0.0],
                        bounds=([0.05, an["kappa"] * 0.5, -3.0, -0.1], [1.05, an["kappa"] * 2.0, 3.0, 0.1]))   # A may be < 1 if W2 != W
    resid = np.sqrt(np.mean((sin2(z[win], *popt) - frac2[win]) ** 2))
    # phasor convention check: phase slope along guide 1 (first half coupling length only)
    iy1 = np.argmin(np.abs(y - r["y1"]))
    ph = np.unwrap(np.angle(r["Ez"][:, iy1]))
    zz = (z > 1.0) & (z < min(an["L_c"] - 1.0, 12.0))
    slope_ph = np.polyfit(x[zz], ph[zz], 1)[0]
    # Meep uses e^{-iωt}: a +x wave has phase increasing with x. Our convention (notes §3) is the conjugate.
    E = np.conj(r["Ez"]) if slope_ph > 0 else r["Ez"]
    n_avg_fdtd = abs(slope_ph) / K0
    # the Poynting guard in fdtd_coupler.run_cw must never fire: <S_x> = -1/2 Re(Ez Hy*) is the +x flux for Ez polarisation
    assert not r["poynting_sign_flipped"], "run_cw flipped the Poynting sign: the stated <S_x> formula would be wrong"
    # second-order Yee numerical dispersion along a grid axis (Taflove): k_num/k - 1 ≈ (kΔx)²/24 · (1 − S²/n²), S = Courant
    n_ref = eig[f"{g*1e3:.0f}nm"]["n_even"] * 0.5 + eig[f"{g*1e3:.0f}nm"]["n_odd"] * 0.5
    yee_pct = float(100 * (n_ref * K0 / RES) ** 2 / 24 * (1 - COURANT ** 2 / n_ref ** 2))
    cw[g] = dict(z=z, y=y, E=E, Sx=r["Sx"], P1=r["P1"] / P0.max(), P2=r["P2"] / P0.max(), P0=P0 / P0.max(),
                 eps=r["eps"], y1=r["y1"], y2=r["y2"], w1=r["w1"], w2=r["w2"], fit=popt, win=win)
    fdtd[f"{g*1e3:.0f}nm"] = dict(
        kappa_fit_per_um=float(popt[1]), L_c_fit_um=float(np.pi / (2 * popt[1])), fit_amplitude=float(popt[0]),
        fit_z0_um=float(popt[2]), fit_offset=float(popt[3]), fit_rms_residual=float(resid),
        kappa_analytic_per_um=float(an["kappa"]), L_c_analytic_um=float(an["L_c"]),
        kappa_meep_eigen_per_um=eig[f"{g*1e3:.0f}nm"]["kappa_per_um"],
        agreement_vs_analytic_pct=float(100 * popt[1] / an["kappa"]),
        agreement_vs_meep_eigen_pct=float(100 * popt[1] / eig[f"{g*1e3:.0f}nm"]["kappa_per_um"]),
        max_fraction_in_guide2=float(frac2[win].max()),
        total_power_variation_pct=float(100 * (P0[win].max() - P0[win].min()) / P0[win].mean()),
        phase_slope_sign=("+" if slope_ph > 0 else "-"), n_avg_from_phase_slope=float(n_avg_fdtd),
        n_avg_expected_analytic=float(0.5 * (an["n_even"] + an["n_odd"])),
        n_avg_expected_meep_eigen=float(0.5 * (eig[f"{g*1e3:.0f}nm"]["n_even"] + eig[f"{g*1e3:.0f}nm"]["n_odd"])),
        n_avg_excess_vs_meep_eigen_pct=float(100 * (n_avg_fdtd / n_ref - 1)),
        yee_dispersion_estimate_pct=yee_pct, poynting_sign_flipped=bool(r["poynting_sign_flipped"]))
    log(f"C FDTD gap {g*1e3:.0f} nm: κ_fit = {popt[1]:.4f}/µm (analytic {an['kappa']:.4f}, Meep eigen "
        f"{eig[f'{g*1e3:.0f}nm']['kappa_per_um']:.4f}); L_c = {np.pi/(2*popt[1]):.2f} µm; max P2/P0 = {frac2[win].max():.3f}; "
        f"n̄ from phase = {n_avg_fdtd:.4f} (expect {0.5*(an['n_even']+an['n_odd']):.4f} analytic, {n_ref:.4f} same-grid eigen; "
        f"Yee estimate +{yee_pct:.2f} %); ΣP varies {fdtd[f'{g*1e3:.0f}nm']['total_power_variation_pct']:.2f} %")
results["fdtd_cw"] = fdtd
results["inputs"]["meep_phasor_note"] = ("Meep DFT phasors follow e^{-iωt} (phase grows along +x); they are conjugated "
                                         "so that the plotted Ẽ obeys the notes' e^{jωt}, e^{-jβz} convention")
timing["C_fdtd_cw_s"] = time.time() - t


def outline_guides(ax, c, lw=0.5, axis="y"):
    """Draw the slab edges of guide 1 and guide 2 from their centres and (possibly different) widths."""
    for yc, w in ((c["y1"], c["w1"]), (c["y2"], c["w2"])):
        if axis == "y":
            ax.axhline(yc - w / 2, color=PALETTE["ink2"], lw=lw); ax.axhline(yc + w / 2, color=PALETTE["ink2"], lw=lw)
        else:
            ax.axvspan(yc - w / 2, yc + w / 2, color=PALETTE["line"], alpha=0.6)


# ---- figure: |E| maps
fig, axs = plt.subplots(len(GAPS_FDTD), 1, figsize=(13, 2.6 * len(GAPS_FDTD)), sharex=True)
for ax, g in zip(axs, GAPS_FDTD):
    c = cw[g]; A = np.abs(c["E"]).T
    ax.imshow(A, origin="lower", extent=[c["z"][0], c["z"][-1], c["y"][0], c["y"][-1]], cmap="Blues", aspect="auto",
              vmin=0, vmax=A.max())
    outline_guides(ax, c)
    Lc = fdtd[f"{g*1e3:.0f}nm"]["L_c_fit_um"]
    for k in range(1, int(34 / Lc) + 1):
        ax.axvline(k * Lc, color=SERIES[1], lw=0.8, ls="--")
    ax.set_ylabel("y (µm)"); ax.set_xlim(-1.5, 34.5); ax.set_ylim(-1.2, 1.2)
    ax.text(-1.3, 0.95, f"gap {g*1e3:.0f} nm:  L_c (fit) = {Lc:.1f} µm", fontsize=9, color=PALETTE["ink"],
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.text(-1.3, -1.1, "guide 2 starts at z = 0 →", fontsize=7.5, color=PALETTE["ink2"])
axs[0].set_title("|E| of the steady-state field at 1310 nm: the light launched in the top guide walks to the bottom guide "
                 "and back every coupling length L_c (dashed)", fontsize=10)
axs[-1].set_xlabel("z along the coupler (µm), z = 0 where the second guide begins")
fig.tight_layout(); fig.savefig(OUT / "fdtd_fields.png"); plt.close(fig)

# ---- figure: power in each guide vs z with fits
fig, axs = plt.subplots(len(GAPS_FDTD), 1, figsize=(13.5, 2.9 * len(GAPS_FDTD)), sharex=True)
for ax, g in zip(axs, GAPS_FDTD):
    c = cw[g]; f = fdtd[f"{g*1e3:.0f}nm"]; z = c["z"]
    ax.plot(z, c["P1"], color=SERIES[0], label="P₁/P₀ guide 1 (FDTD Poynting flux, y > 0)")
    ax.plot(z, c["P2"], color=SERIES[1], label="P₂/P₀ guide 2 (y < 0)")
    ax.plot(z, c["P0"], color=PALETTE["muted"], lw=1, label="P₁ + P₂ (constant: lossless)")
    zz = np.linspace(0, 35, 400)
    ax.plot(zz, sin2(zz, *c["fit"]), "--", color=SERIES[3], lw=1.3,
            label=f"fit sin²(κ_c z): κ_c = {f['kappa_fit_per_um']:.4f}/µm, L_c = {f['L_c_fit_um']:.2f} µm")
    ax.plot(zz, np.cos(f["kappa_analytic_per_um"] * zz) ** 2, ":", color=SERIES[4], lw=1.3,
            label=f"analytic cos²((β₊−β₋)z/2): κ_c = {f['kappa_analytic_per_um']:.4f}/µm, L_c = {f['L_c_analytic_um']:.2f} µm")
    ax.set_ylabel("power fraction"); ax.set_ylim(-0.05, 1.12); ax.set_xlim(-1.5, 34.5)
    ax.set_title(f"gap {g*1e3:.0f} nm: P₁ = cos²(κ_c z), P₂ = sin²(κ_c z), P₁ + P₂ = 1 (notes §27)", fontsize=9.5)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)   # outside the axes: never over the curves
axs[-1].set_xlabel("z along the coupler (µm)")
fig.tight_layout(); fig.savefig(OUT / "power_vs_z.png", bbox_inches="tight"); plt.close(fig)

# ---- figure: supermodes and the beat (gap 200 nm)
g = 0.20; c = cw[g]; an = sc.supermodes(g, REF.n_si, REF.n_sio2, 0.22, LAM)
yy = np.linspace(-1.5, 1.5, 1201)
layers = an["layers"]; x_start = -(0.22 + g / 2)
F_even = sc.te_profile(an["n_even"], layers, REF.n_sio2, LAM, yy, x_start)
F_odd = sc.te_profile(an["n_odd"], layers, REF.n_sio2, LAM, yy, x_start)
if F_odd[yy > 0].mean() < 0:
    F_odd = -F_odd
# normalise both supermodes to the same power ∫F² dy so that (F+ ± F-)/2 are the single-guide fields
F_even /= np.sqrt(np.trapezoid(F_even ** 2, yy)); F_odd /= np.sqrt(np.trapezoid(F_odd ** 2, yy))
me, mo = eig_prof["even"], eig_prof["odd"]
me_F = me["F"] / np.sqrt(np.trapezoid(me["F"] ** 2, me["y"])); mo_F = mo["F"] / np.sqrt(np.trapezoid(mo["F"] ** 2, mo["y"]))
if mo_F[mo["y"] > 0].mean() < 0: mo_F = -mo_F
if me_F.mean() < 0: me_F = -me_F
zb = np.linspace(0, 2 * an["L_c"], 600)
beat = np.abs(0.5 * (F_even[None, :] * np.exp(-1j * an["beta_even"] * zb[:, None]) +
                     F_odd[None, :] * np.exp(-1j * an["beta_odd"] * zb[:, None]))) ** 2
fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], height_ratios=[1, 1])
ax = fig.add_subplot(gs[:, 0])
ax.plot(yy, F_even, color=SERIES[0], label=f"even supermode F₊, n₊ = {an['n_even']:.4f}")
ax.plot(yy, F_odd, color=SERIES[1], label=f"odd supermode F₋, n₋ = {an['n_odd']:.4f}")
ax.plot(me["y"], me_F, "o", ms=2.5, markevery=3, color=SERIES[0], alpha=0.6, label=f"Meep eigenmode, n₊ = {me['n_eff']:.4f}")
ax.plot(mo["y"], mo_F, "s", ms=2.5, markevery=3, color=SERIES[1], alpha=0.6, label=f"Meep eigenmode, n₋ = {mo['n_eff']:.4f}")
ax.plot(yy, 0.5 * (F_even + F_odd), "--", color=SERIES[2], label="(F₊ + F₋)/2 = light in guide 1 (z = 0)")
ax.plot(yy, 0.5 * (F_even - F_odd), ":", color=SERIES[3], label="(F₊ − F₋)/2 = light in guide 2 (z = L_c)")
outline_guides(ax, c, axis="x")
ax.set_xlabel("y across the two slabs (µm)"); ax.set_ylabel("field profile (normalised)"); ax.set_xlim(-1.2, 1.2)
ax.set_title("Two 220 nm slabs, 200 nm gap: even and odd supermodes;\ntheir sum and difference are the single-guide fields", fontsize=9.5)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
ax = fig.add_subplot(gs[0, 1])
ax.imshow(beat.T, origin="lower", extent=[zb[0], zb[-1], yy[0], yy[-1]], cmap="Blues", aspect="auto")
ax.set_ylim(-1.2, 1.2); ax.set_ylabel("y (µm)")
ax.set_title(f"Analytic beat |½F₊e^(−jβ₊z) + ½F₋e^(−jβ₋z)|²: the two supermodes drift out of phase and the power moves; "
             f"L_c = π/(β₊−β₋) = {an['L_c']:.2f} µm", fontsize=8.5)
ax = fig.add_subplot(gs[1, 1])
zsel = (c["z"] >= 0) & (c["z"] <= 2 * an["L_c"])
ax.imshow((np.abs(c["E"][zsel, :]) ** 2).T, origin="lower", extent=[c["z"][zsel][0], c["z"][zsel][-1], c["y"][0], c["y"][-1]],
          cmap="Blues", aspect="auto")
ax.set_ylim(-1.2, 1.2); ax.set_ylabel("y (µm)"); ax.set_xlabel("z (µm)")
ax.set_title(f"Meep FDTD |E|² over the same 2 L_c: same picture; fitted L_c = {fdtd['200nm']['L_c_fit_um']:.2f} µm", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "supermode_beat.png", bbox_inches="tight"); plt.close(fig)

# ---- figure: κ vs gap (log axis)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.semilogy(gaps_fine * 1e3, kappa_tm, color=SERIES[0], label="analytic supermodes: κ_c = (β₊ − β₋)/2 (5-layer transfer matrix)")
ax.semilogy(gaps_fine * 1e3, kappa_cmt, "--", color=SERIES[1], label="coupled-mode closed form (overlap of the two tails)")
ax.semilogy([g * 1e3 for g in GAPS_EIG], [eig[f'{g*1e3:.0f}nm']['kappa_per_um'] for g in GAPS_EIG], "s", ms=6, color=SERIES[2],
            label="Meep eigenmode solver on the FDTD grid")
ax.semilogy([g * 1e3 for g in GAPS_FDTD], [fdtd[f'{g*1e3:.0f}nm']['kappa_fit_per_um'] for g in GAPS_FDTD], "o", ms=9,
            mfc="none", mew=2, color=SERIES[3], label="Meep FDTD: fit of P₂(z) = sin²(κ_c z)")
gref = np.array([0.10, 0.45])
ax.semilogy(gref * 1e3, np.exp(icpt + slope * gref), ":", color=PALETTE["ink2"], lw=1.2,
            label=f"e^(−γ·gap) with γ = {-slope:.2f}/µm from the slope (single-slab γ = {slab['gamma']:.2f}/µm, 1/γ = {1e3/slab['gamma']:.0f} nm)")
for g in GAPS_FDTD:   # coupling length at each FDTD point (L_c = π/(2κ_c); one axis, no unit-transform twin)
    f = fdtd[f"{g*1e3:.0f}nm"]
    ax.annotate(f"L_c = {f['L_c_fit_um']:.1f} µm", xy=(g * 1e3, f["kappa_fit_per_um"]), xytext=(12, 4), textcoords="offset points",
                fontsize=7.5, color=SERIES[3])
ax.set_xlabel("gap between the slabs (nm)"); ax.set_ylabel("coupling coefficient κ_c (rad/µm)")
ax.set_title("κ_c falls exponentially with the gap, with the evanescent decay length of one slab:\n10 nm more gap = 12 % less κ_c (2-D slab)", fontsize=10)
ax.legend(fontsize=7, loc="lower left"); ax.set_xlim(50, 500)
fig.tight_layout(); fig.savefig(OUT / "kappa_vs_gap.png"); plt.close(fig)
log("C figures written")

# ================================================================== D. pulse video
t = time.time()
g = 0.20
times, frames, xp, yp, epsp = fc.run_pulse(g, RES, fwidth_frac=0.2, n_frames=240, dt_frame=0.7, downsample=2)
zp = xp - fc.X_START
vmax = 0.5 * np.abs(frames).max()
top = yp > 0
# pulse energy per guide: the instantaneous ∫E² dy oscillates at 2ω (a comb under the envelope), so use the envelope
# |E_a|² of the analytic signal E_a = E + jH{E} (Hilbert transform along z, frame by frame), integrated over each half-plane
env2 = np.abs(hilbert(frames.astype(np.float64), axis=1)) ** 2
e1 = env2[:, :, top].sum(axis=2); e2 = env2[:, :, ~top].sum(axis=2)
emax = max(e1.max(), e2.max())
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.4), gridspec_kw=dict(height_ratios=[1.6, 1]), sharex=True)
im = ax1.imshow(frames[0].T, origin="lower", extent=[zp[0], zp[-1], yp[0], yp[-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
outline_guides(ax1, c)
ax1.set_ylim(-1.2, 1.2); ax1.set_ylabel("y (µm)")
Lc = fdtd["200nm"]["L_c_fit_um"]


def pulse_title(i):
    return (f"A 1310 nm pulse launched in guide 1 (gap 200 nm) hops to guide 2 at z = L_c = {Lc:.1f} µm and back at 2 L_c\n"
            f"t = {times[i] * 1e3 / 299.792458:.0f} fs")


ttl = ax1.set_title(pulse_title(0), fontsize=10)      # final-length placeholder so tight_layout reserves the room
l1, = ax2.plot(zp, e1[0] / emax, color=SERIES[0], label="pulse energy envelope in guide 1 (∫|E_env|² dy, y > 0; Hilbert envelope along z)")
l2, = ax2.plot(zp, e2[0] / emax, color=SERIES[1], label="pulse energy envelope in guide 2 (y < 0)")
for k in range(1, 3):
    ax2.axvline(k * Lc, color=SERIES[1], lw=0.8, ls="--")
ax2.text(Lc, 0.45, " L_c", fontsize=8, color=SERIES[1]); ax2.text(2 * Lc, 0.45, " 2 L_c", fontsize=8, color=SERIES[1])   # below the legend
ax2.set_ylim(0, 1.05); ax2.set_xlim(-1.5, 34.5); ax2.set_xlabel("z along the coupler (µm)"); ax2.set_ylabel("envelope energy (norm.)")
ax2.legend(fontsize=8, loc="upper right")
fig.tight_layout()


def upd(i):
    im.set_data(frames[i].T); l1.set_ydata(e1[i] / emax); l2.set_ydata(e2[i] / emax)
    ttl.set_text(pulse_title(i))
    return im, l1, l2, ttl


anim = FuncAnimation(fig, upd, frames=len(frames), blit=False)
anim.save(OUT / "pulse_hopping.mp4", writer=FFMpegWriter(fps=30, bitrate=2500))
plt.close(fig)
# contact sheet
first = int(np.argmax(np.abs(frames).max(axis=(1, 2)) > 0.5 * np.abs(frames).max()))
idx = np.linspace(first, len(frames) - 30, 6).astype(int)
fig, axs = plt.subplots(6, 1, figsize=(12, 9), sharex=True)
for ax, i in zip(axs, idx):
    ax.imshow(frames[i].T, origin="lower", extent=[zp[0], zp[-1], yp[0], yp[-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    outline_guides(ax, c, lw=0.4)
    ax.set_ylim(-1.2, 1.2); ax.set_xlim(-1.5, 34.5); ax.set_ylabel("y (µm)")
    ax.text(-1.2, 0.85, f"t = {times[i] * 1e3 / 299.792458:.0f} fs", fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
axs[0].set_title("Stills from pulse_hopping.mp4: E_z of a pulse crossing from the top guide to the bottom one and back (gap 200 nm)", fontsize=10)
axs[-1].set_xlabel("z along the coupler (µm)")
fig.tight_layout(); fig.savefig(OUT / "pulse_hopping_frames.png"); plt.close(fig)
results["pulse_video"] = dict(gap_nm=200, n_frames=int(len(frames)), duration_s=len(frames) / 30,
                              sim_time_fs=float(times[-1] * 1e3 / 299.792458))
timing["D_pulse_video_s"] = time.time() - t
log(f"D pulse video: {len(frames)} frames, {len(frames)/30:.1f} s")

# ================================================================== E. CW phase video (zoom at the 50/50 point)
t = time.time()
c = cw[0.20]; Lc = fdtd["200nm"]["L_c_fit_um"]
z0 = Lc / 2
zsel = (c["z"] > z0 - 3) & (c["z"] < z0 + 3)
Ez = c["E"][zsel, :]; zz = c["z"][zsel]
vmax = np.abs(Ez).max()
n_fr = 72; periods = 2
fig, ax = plt.subplots(figsize=(9, 4.2))
im = ax.imshow(np.real(Ez).T, origin="lower", extent=[zz[0], zz[-1], c["y"][0], c["y"][-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
outline_guides(ax, c)
ax.set_ylim(-1.2, 1.2); ax.set_xlabel("z (µm)"); ax.set_ylabel("y (µm)")
Tper = LAM / 0.299792458   # optical period in fs (4.37 fs)


def cw_title(i):
    return (f"Re{{Ẽ(y,z) e^(jωt)}} around the 50/50 point (z ≈ L_c/2 = {z0:.1f} µm, gap 200 nm): the phase fronts advance +z\n"
            f"every λ_g ≈ {1e3 * LAM / fdtd['200nm']['n_avg_from_phase_slope']:.0f} nm while the envelope stays;  t = {Tper * periods * i / n_fr:.2f} fs")


ttl = ax.set_title(cw_title(0), fontsize=8.5)         # final-length placeholder so tight_layout reserves the room
fig.tight_layout()


def upd2(i):
    ph = 2 * np.pi * periods * i / n_fr
    im.set_data(np.real(Ez * np.exp(1j * ph)).T)
    ttl.set_text(cw_title(i))
    return im, ttl


anim = FuncAnimation(fig, upd2, frames=n_fr, blit=False)
anim.save(OUT / "cw_phase_zoom.mp4", writer=FFMpegWriter(fps=30, bitrate=2500))
plt.close(fig)
fig, axs = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True, sharey=True)
for k, ax in enumerate(axs.ravel()):
    ph = 2 * np.pi * k / 4
    ax.imshow(np.real(Ez * np.exp(1j * ph)).T, origin="lower", extent=[zz[0], zz[-1], c["y"][0], c["y"][-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    outline_guides(ax, c)
    ax.set_ylim(-1.2, 1.2); ax.set_title(f"ωt = {k * 90}°  (t = {Tper * k / 4:.2f} fs)", fontsize=9)
for ax in axs[-1]: ax.set_xlabel("z (µm)")
for ax in axs[:, 0]: ax.set_ylabel("y (µm)")
fig.suptitle("Stills from cw_phase_zoom.mp4: the standing pattern across y travels along z (notes §21); both guides carry equal power here", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "cw_phase_zoom_frames.png"); plt.close(fig)
timing["E_cw_phase_video_s"] = time.time() - t
log("E CW phase video written")

# ================================================================== F, G. SAX and gdsfactory in .venv
for tag, script in (("F_sax_s", "sax_part.py"), ("G_gds_s", "gds_part.py")):
    t = time.time()
    proc = subprocess.run([str(VENV_PY), script], cwd=HERE, capture_output=True, text=True)
    (OUT / f"{script[:-3]}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout); print(proc.stderr)
        raise SystemExit(f"{script} failed (see out/{script[:-3]}.log)")
    for line in proc.stdout.splitlines():
        if line.startswith("["):
            log("   " + line)
    timing[tag] = time.time() - t
results["sax"] = json.loads((OUT / "sax_results.json").read_text())
results["gds"] = json.loads((OUT / "gds_results.json").read_text())

# ================================================================== H. capstone numbers + results.json
g_text = K0 * np.sqrt(REF.neff ** 2 - REF.n_sio2 ** 2)
cap = results["sax"]["point_coupler"]
# which 2-D slab gap would give the capstone κ₀ = asin(√κ²)/L_eff?  (i) exact: invert the analytic κ_c(gap) table,
# (ii) exponential estimate from the 200 nm point: gap = 0.2 − ln(κ₀/κ_c(200 nm))/γ
kappa0 = cap["textbook_neff2p5"]["kappa_c_for_ref_kappa2_per_um"]
gap_for_kappa0_exact = float(np.interp(-np.log(kappa0), -np.log(kappa_tm), gaps_fine))       # ln κ is monotone in gap
gap_for_kappa0_exp = float(0.20 - np.log(kappa0 / results["kappa_vs_gap_analytic"]["per_gap"]["200nm"]["kappa_per_um"]) / slab["gamma"])
results["capstone"] = dict(
    kappa2_reference=REF.kappa2, t_reference=float(np.sqrt(1 - REF.kappa2)), a_reference=REF.a_round_trip,
    a_used_sax=results["sax"]["round_trip"]["a_used"], loss_db_cm_used_sax=results["sax"]["round_trip"]["loss_db_cm_used"],
    Tmin_at_pm10nm_sax_circuit={k: results["sax"]["ring_spectra_textbook_gamma"][k]["T_min"] for k in ("-10", "+0", "+10")},
    gamma_textbook_per_um=float(g_text), decay_length_textbook_nm=float(1e3 / g_text),
    L_eff_textbook_um=cap["textbook_neff2p5"]["L_eff_um"], L_eff_slab2d_um=cap["slab2d_neff2p99"]["L_eff_um"],
    kappa2_change_pct_per_10nm_textbook=cap["textbook_neff2p5"]["kappa2_pct_change_per_10nm"],
    kappa2_change_pct_per_10nm_slab2d=cap["slab2d_neff2p99"]["kappa2_pct_change_per_10nm"],
    small_signal_2gamma_dg_pct_textbook=cap["textbook_neff2p5"]["small_signal_pct_per_10nm"],
    small_signal_2gamma_dg_pct_slab2d=cap["slab2d_neff2p99"]["small_signal_pct_per_10nm"],
    Tmin_at_pm10nm_textbook=cap["textbook_neff2p5"]["Tmin_at_gap_err_nm"],
    kappa0_for_ref_kappa2_per_um=float(kappa0),
    gap_for_kappa0_slab2d_nm=1e3 * gap_for_kappa0_exact, gap_for_kappa0_slab2d_exp_estimate_nm=1e3 * gap_for_kappa0_exp,
    gap_error_nm_for_ref_Tmin_textbook=cap["textbook_neff2p5"]["gap_error_nm_for_ref_Tmin"],
    gap_error_nm_for_ref_Tmin_slab2d=cap["slab2d_neff2p99"]["gap_error_nm_for_ref_Tmin"],
    yee_dispersion_estimate_pct_200nm=fdtd["200nm"]["yee_dispersion_estimate_pct"],
    n_avg_excess_vs_meep_eigen_pct_200nm=fdtd["200nm"]["n_avg_excess_vs_meep_eigen_pct"],
    poynting_sign_flipped_any=any(fdtd[k]["poynting_sign_flipped"] for k in fdtd),
    ring_fsr_nm_sax=results["sax"]["ring_spectra_textbook_gamma"]["fsr_nm"], ring_fsr_nm_expected=REF.fsr_nm,
    ring_fwhm_pm_sax=results["sax"]["ring_spectra_textbook_gamma"]["+0"]["fwhm_pm"], ring_fwhm_pm_expected=REF.fwhm_pm,
    ring_fwhm_pm_analytic_same_a=results["sax"]["ring_spectra_textbook_gamma"]["fwhm_expected_pm_analytic"])
timing["total_s"] = time.time() - T0
results["timing_s"] = timing
results["versions"] = dict(meep=mp.__version__, numpy=np.__version__, sax=results["sax"]["versions"]["sax"],
                           gdsfactory=results["gds"]["versions"]["gdsfactory"])
results["files"] = sorted(p.name for p in OUT.iterdir())
(OUT / "results.json").write_text(json.dumps(results, indent=2))

# ---- results.txt (human summary)
f2 = fdtd["200nm"]; a2 = results["kappa_vs_gap_analytic"]["per_gap"]["200nm"]
lines = [
    "Experiment 11: directional coupler, headline numbers",
    f"single 220 nm slab (analytic): n_eff = {slab['n_eff']:.4f}, γ = {slab['gamma']:.3f}/µm, 1/γ = {1e3/slab['gamma']:.1f} nm; Meep eigenmode n_eff = {n_single_meep:.4f}",
    f"gap 200 nm: n+ = {a2['n_even']:.4f}, n- = {a2['n_odd']:.4f} (analytic); Meep eigen n+ = {eig['200nm']['n_even']:.4f}, n- = {eig['200nm']['n_odd']:.4f}",
    f"gap 200 nm: κ_c analytic {a2['kappa_per_um']:.4f}/µm, CMT {a2['kappa_cmt_per_um']:.4f}/µm, Meep eigen {eig['200nm']['kappa_per_um']:.4f}/µm, FDTD fit {f2['kappa_fit_per_um']:.4f}/µm ({f2['agreement_vs_analytic_pct']:.1f} % of analytic)",
    f"gap 200 nm: L_c analytic {a2['L_c_um']:.2f} µm, FDTD {f2['L_c_fit_um']:.2f} µm",
]
for g in GAPS_FDTD:
    k = f"{g*1e3:.0f}nm"; f = fdtd[k]
    lines.append(f"gap {k}: κ_fdtd = {f['kappa_fit_per_um']:.4f}/µm vs analytic {f['kappa_analytic_per_um']:.4f} ({f['agreement_vs_analytic_pct']:.1f} %), "
                 f"L_c = {f['L_c_fit_um']:.2f} µm, max P2/P0 = {f['max_fraction_in_guide2']:.3f}")
lines += [
    f"κ_c vs gap: ln-slope γ = {-slope:.3f}/µm vs single-slab γ {slab['gamma']:.3f}/µm ({results['kappa_vs_gap_analytic']['gamma_fit_agreement_pct']:.1f} %)",
    f"SAX: max |S†S − I| = {results['sax']['unitarity']['max_abs_deviation_SdagS_from_I']:.1e}; cross phase {results['sax']['unitarity']['cross_phase_deg']:.0f}°",
    f"point coupler L_eff = √(2πR/γ): {cap['textbook_neff2p5']['L_eff_um']:.2f} µm (textbook γ), {cap['slab2d_neff2p99']['L_eff_um']:.2f} µm (2-D slab γ)",
    f"κ² = 10.7 % ±10 nm gap: {cap['textbook_neff2p5']['kappa2_pct_change_per_10nm'][0]:+.0f} % / {cap['textbook_neff2p5']['kappa2_pct_change_per_10nm'][1]:+.0f} % (textbook γ; small-signal 2γΔg = {cap['textbook_neff2p5']['small_signal_pct_per_10nm']:.0f} %)",
    f"SAX ring: FSR {results['capstone']['ring_fsr_nm_sax']:.2f} nm (expect {REF.fsr_nm}), FWHM {results['capstone']['ring_fwhm_pm_sax']:.0f} pm (expect {REF.fwhm_pm:.0f})",
    f"capstone κ₀ = {kappa0:.4f} rad/µm = κ_c of a {1e3*gap_for_kappa0_exact:.0f} nm gap in the 2-D slab model (exponential estimate {1e3*gap_for_kappa0_exp:.0f} nm)",
    f"reference T_min = {REF.t_min} corresponds to a gap error of {cap['textbook_neff2p5']['gap_error_nm_for_ref_Tmin']['over_coupled_smaller_gap']:+.1f} / "
    f"{cap['textbook_neff2p5']['gap_error_nm_for_ref_Tmin']['under_coupled_larger_gap']:+.1f} nm (textbook γ)",
    f"FDTD n̄ (200 nm) exceeds the same-grid eigenmode n̄ by {fdtd['200nm']['n_avg_excess_vs_meep_eigen_pct']:.2f} %; Yee dispersion estimate {fdtd['200nm']['yee_dispersion_estimate_pct']:.2f} %",
    f"total run time {timing['total_s']:.0f} s",
]
(OUT / "results.txt").write_text("\n".join(lines) + "\n")

# ---- tools.json
tools = [
    dict(tool="Meep (pymeep) 2-D FDTD", version=mp.__version__,
         what_it_is="Open-source finite-difference time-domain solver for Maxwell's equations (MIT). It steps E and H forward in time on a Yee grid, so any geometry and any source can be simulated without approximations beyond the grid.",
         used_for="Two 220 nm silicon slabs in silica at 1310 nm with gaps 150/200/300 nm, launched with the single-slab eigenmode in guide 1 (guide 2 starts at z = 0). Steady-state DFT fields Ez, Hy give the time-averaged Poynting flux along the propagation axis, <S_x> = -1/2 Re(Ez Hy*) (Meep x = the notes' z), integrated over each half-plane to get P1(z), P2(z); the code records that this formula never needed a sign flip (poynting_sign_flipped = false). A pulse run gives the video of power hopping (Hilbert envelope energy per guide underneath).",
         result=f"P2/P0 = sin²(κ_c z) with κ_c = {fdtd['150nm']['kappa_fit_per_um']:.4f}, {fdtd['200nm']['kappa_fit_per_um']:.4f}, {fdtd['300nm']['kappa_fit_per_um']:.4f} rad/µm for 150/200/300 nm ({fdtd['150nm']['agreement_vs_analytic_pct']:.1f} %, {fdtd['200nm']['agreement_vs_analytic_pct']:.1f} %, {fdtd['300nm']['agreement_vs_analytic_pct']:.1f} % of the analytic (β+−β−)/2); L_c(200 nm) = {fdtd['200nm']['L_c_fit_um']:.2f} µm vs {a2['L_c_um']:.2f} µm analytic; P1 + P2 constant to {fdtd['200nm']['total_power_variation_pct']:.2f} %.",
         how_to_observe="cd experiments/11_directional_coupler && ../../.meep/bin/python run.py (about 2 to 3 min on this laptop; the measured value is timing_s.total_s in out/results.json); look at out/fdtd_fields.png (|E| maps, dashed lines every L_c), out/power_vs_z.png (P1, P2 with the sin² fit), out/pulse_hopping.mp4 and out/cw_phase_zoom.mp4. Change GAPS_FDTD, RES or LAM at the top of run.py (K0 and fdtd_coupler follow LAM), SX/X_START in fdtd_coupler.py, or pass W2 to fc.run_cw for non-identical guides (the fitted rate is then sqrt(kappa^2 + dbeta^2/4)), and re-run."),
    dict(tool="Meep eigenmode solver (MPB inside Meep)", version=mp.__version__,
         what_it_is="Meep embeds the MPB plane-wave eigenmode solver: given a cross-section of the FDTD grid it returns the guided modes (β, field profile) at a fixed frequency. It is what EigenModeSource and mode decomposition use internally.",
         used_for="β of the single slab and β+ (even, EVEN_Y) and β− (odd, ODD_Y) of the coupled slabs at six gaps, on the same 40 px/µm grid as the FDTD, plus the two supermode profiles at 200 nm gap.",
         result=f"Single slab n_eff = {n_single_meep:.4f} vs analytic {slab['n_eff']:.4f} ({results['meep_eigenmode']['single_slab_vs_analytic_pct']:.1f} %, the grid under-resolves the 220 nm core); κ_c = (β+−β−)/2 at 200 nm = {eig['200nm']['kappa_per_um']:.4f} rad/µm ({eig['200nm']['kappa_vs_analytic_pct']:.1f} % of analytic); the FDTD fit agrees with this same-grid value to {fdtd['200nm']['agreement_vs_meep_eigen_pct']:.1f} %.",
         how_to_observe="Run run.py; see out/supermode_beat.png (left: profiles with Meep markers) and out/kappa_vs_gap.png (square markers). Edit GAPS_EIG in run.py; fdtd_coupler.eigen_supermodes() is the call."),
    dict(tool="numpy/scipy transfer-matrix slab solver (slab_coupler.py)", version=np.__version__,
         what_it_is="Hand-written solver: the TE slab eigenvalue equation h tan(hd) = γ (notes §19) with scipy.optimize.brentq, and a 2x2 transfer matrix for (F, F') across a stack of layers (notes §18) whose bound-state condition gives all TE modes of any multilayer.",
         used_for="Reference numbers: single-slab n_eff, γ, decay length; even/odd supermodes of the two-slab stack at any gap; κ_c = (β+−β−)/2; the Yariv coupled-mode closed form κ_c = 2h²γe^{-γs}/[β(W+2/γ)(h²+γ²)]; the point-coupler length √(2πR/γ).",
         result=f"n_eff = {slab['n_eff']:.4f}, 1/γ = {1e3/slab['gamma']:.1f} nm; κ_c(200 nm) = {a2['kappa_per_um']:.4f} rad/µm, L_c = {a2['L_c_um']:.2f} µm; CMT within {results['kappa_vs_gap_analytic']['cmt_vs_supermode_max_rel_dev_pct']:.2f} % of the exact supermode splitting; ln κ_c vs gap slope = {-slope:.2f}/µm = {results['kappa_vs_gap_analytic']['gamma_fit_agreement_pct']:.1f} % of the single-slab γ.",
         how_to_observe="out/kappa_vs_gap.png (solid and dashed lines) and out/supermode_beat.png; import slab_coupler in either interpreter and call supermodes(gap) or cmt_kappa(gap)."),
    dict(tool="SAX", version=results["sax"]["versions"]["sax"],
         what_it_is="JAX-based S-parameter circuit simulator from the gdsfactory ecosystem: photonic components are Python functions returning S-dictionaries, and sax.circuit connects them into a netlist and solves the composite S-matrix (differentiable, vectorised over wavelength or parameters).",
         used_for="The 2x2 coupler model S = [[t, −jK], [−jK, t]] with t = cos(κ_c L); unitarity and reciprocity check; |t|², |K|² vs length; the interference of two coherent inputs vs relative phase (directly and as a Mach-Zehnder circuit); the all-pass ring circuit (coupler + 39.6 µm doped waveguide) to show what a ±10 nm gap error does to the notch.",
         result=f"max |S†S − I| = {results['sax']['unitarity']['max_abs_deviation_SdagS_from_I']:.0e}; cross-port phase −90°; two ½ W inputs give 0 to 1 W at either port depending on φ, total always 1 W; ring FSR {results['capstone']['ring_fsr_nm_sax']:.2f} nm (expect {REF.fsr_nm}), FWHM {results['capstone']['ring_fwhm_pm_sax']:.0f} pm (expect {REF.fwhm_pm:.0f}); κ² = 0.107 → 0.129 / 0.089 for ∓10 nm gap (textbook γ), T_min from 0 to {results['sax']['ring_spectra_textbook_gamma']['-10']['T_min']:.4f} / {results['sax']['ring_spectra_textbook_gamma']['+10']['T_min']:.4f} (SAX circuit; closed form {cap['textbook_neff2p5']['Tmin_at_gap_err_nm']['-10']:.4f} / {cap['textbook_neff2p5']['Tmin_at_gap_err_nm']['+10']:.4f} with the same a = {results['sax']['round_trip']['a_used']:.4f}).",
         how_to_observe="../../.venv/bin/python sax_part.py (run.py also calls it); out/sax_coupler_vs_length.png, out/sax_interference.png, out/critical_coupling_lottery.png, out/sax_results.json. Change A_RT (one round-trip retention feeds both the closed form and the waveguide loss), the gap-error list dg, or REF.kappa2 usage in sax_part.py."),
    dict(tool="gdsfactory", version=results["gds"]["versions"]["gdsfactory"],
         what_it_is="Python layout generator for photonic chips (built on KLayout): parametric components (rings, couplers, bends) with ports and cross-sections, exported to GDSII for fabrication.",
         used_for="ring_single (R = 6.3 µm circular bends, 500 nm strips, 200 nm gap to a bus) and coupler (straight directional coupler, 200 nm gap, parallel length = L_c of the 200 nm gap) exported to GDS; polygons re-read to measure the gap and radius and to verify the parabola gap(z) = g0 + z²/2R used for the point-coupler length.",
         result=f"Measured gap {results['gds']['ring_bus']['gap_nm_measured']:.0f} nm, centreline R {results['gds']['ring_bus']['radius_um_centreline']:.2f} µm; L_eff = √(2πR/γ) = {cap['textbook_neff2p5']['L_eff_um']:.2f} µm (textbook γ) / {cap['slab2d_neff2p99']['L_eff_um']:.2f} µm (2-D slab γ), numeric integral over the exact circle agrees to {results['gds']['point_coupler']['textbook_neff2p5']['agreement_pct']:.2f} %.",
         how_to_observe="../../.venv/bin/python gds_part.py; open out/ring_bus_coupler.gds / out/directional_coupler.gds in KLayout, or look at out/gds_layout.png and out/point_coupler_geometry.png. Change R, GAP in gds_part.py."),
    dict(tool="matplotlib + ffmpeg", version=matplotlib.__version__,
         what_it_is="matplotlib draws every figure and, through FuncAnimation + FFMpegWriter, streams frames to ffmpeg (Homebrew) to encode H.264 mp4 videos.",
         used_for="All PNG figures, the two videos pulse_hopping.mp4 (240 frames, 8 s) and cw_phase_zoom.mp4 (72 frames, 2.4 s) and their contact sheets.",
         result="out/pulse_hopping.mp4, out/cw_phase_zoom.mp4, *_frames.png contact sheets.",
         how_to_observe="open out/pulse_hopping.mp4; frame count and dt_frame are arguments of fc.run_pulse in run.py."),
]
(OUT / "tools.json").write_text(json.dumps(tools, indent=2))
log(f"H done: results.json, tools.json, results.txt; total {timing['total_s']:.0f} s")
print("\n".join(lines))
