#!/usr/bin/env python
"""08. The real 500 x 220 nm SOI strip at 1310 nm, solved with two full-vector mode solvers.

Everything in out/ is regenerated from scratch by this script (Tidy3D local FDFD mode
solver + femwell FEM, both in the .venv interpreter).  Run from this directory:

    cd experiments/08_soi_strip_mode_solvers && ../../.venv/bin/python run.py

Sections (search for "# ---" markers):
  A  reference strip: n_eff, n_g, TE fraction, three confinement numbers, mode fields
  B  group index decomposition: waveguide dispersion + material dispersion -> FSR
  C  width sweep 300-700 nm: single-mode cutoff, width sensitivity
  D  wavelength sweep 1260-1360 nm across the 8 x 200 GHz channel plan
  E  thermo-optic: perturb n_Si and n_SiO2 by dn/dT * dT, recompute n_eff -> dlambda_r/dT
  F  n_eff versus n_g: which capstone numbers depend on which
  G  video of the mode morphing with width
  H  results.json / tools.json / results.txt
"""
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle

from common import PALETTE, REF, SERIES, use_style
from common.params import C0
from materials import group_index, n_si, n_sio2
from solvers import XS, YS, femwell_mesh, find_mode, solve_femwell, solve_tidy3d

OUT = HERE / "out"
if OUT.exists():
    shutil.rmtree(OUT)          # stale outputs from an interrupted run are never kept
OUT.mkdir()
use_style()
plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"

T0 = time.time()
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def pct(a, b):
    """percent difference of a relative to b"""
    return 100.0 * (a - b) / b


# ------------------------------------------------------------------ inputs (all from REF)
LAM = REF.lambda_nm * 1e-3        # um
W, H = REF.wg_width_um, REF.wg_height_um
NSI, NSIO2 = REF.n_si, REF.n_sio2
L_RT = REF.round_trip_um
K0 = 2 * np.pi / LAM
TD_DL = 0.01                      # Tidy3D uniform grid, um (converged: 0.02 -> 0.01 moves n_eff by 4e-4)
TD_BOX = (2.61, 2.01)             # Tidy3D mode plane, um; odd multiple of TD_DL so core edges land mid-cell (see solvers.py)
FEM_RES_REF = 0.01                # femwell core mesh size for the reference solve, um
FEM_RES_SWEEP = 0.02              # femwell core mesh size in sweeps (n_eff converged to 1e-5)

say(f"Inputs: {W*1e3:.0f} x {H*1e3:.0f} nm Si strip (n={NSI}) in SiO2 (n={NSIO2}) at {LAM*1e3:.0f} nm")
say(f"k0 = {K0:.4f} rad/um; textbook REF: n_eff={REF.neff}, n_g={REF.ng}, Gamma={REF.confinement}")

# ================================================================== A. reference strip
say("\n--- A. reference strip, both solvers")
t = time.time()
td_ref = solve_tidy3d(W, H, LAM, NSI, NSIO2, num_modes=4, dl=TD_DL, box=TD_BOX, want_fields=True)
say(f"Tidy3D: {len(td_ref)} guided modes in {time.time()-t:.1f} s")
t = time.time()
fm_ref = femwell_mesh(W, H, resolution=FEM_RES_REF)
WALL_X = np.array([0.0, 0.20, 0.24, W / 2 - 1e-4, W / 2 + 1e-4, 0.26, 0.30, 0.40])
fw_ref = solve_femwell(fm_ref, LAM, NSI, NSIO2, num_modes=4, want_fields=True, cut_x=WALL_X)
say(f"femwell: {len(fw_ref)} guided modes in {time.time()-t:.1f} s ({fm_ref.nelements} triangles)")

for m in td_ref + fw_ref:
    say(f"  {m.solver:8s} {m.label}: n_eff={m.n_eff:.5f} TE={m.te_fraction:.3f} "
        f"Gamma_P={m.gamma_P:.4f} Gamma_E={m.gamma_E:.4f} S_Si={m.S_core:.4f} S_SiO2={m.S_clad:.4f}"
        + (f" n_g,wg={m.n_g_wg:.4f}" if m.n_g_wg else ""))

te0_td, te0_fw = find_mode(td_ref, "TE0"), find_mode(fw_ref, "TE0")
neff_ref = 0.5 * (te0_td.n_eff + te0_fw.n_eff)
gamma_ref = 1.0 / (K0 * np.sqrt(neff_ref**2 - NSIO2**2))       # notes 16: 1/gamma decay length
say(f"TE0 n_eff: Tidy3D {te0_td.n_eff:.4f}, femwell {te0_fw.n_eff:.4f} "
    f"(solver spread {pct(te0_td.n_eff, te0_fw.n_eff):+.2f} %); textbook 2.5 is {pct(REF.neff, neff_ref):+.1f} % lower")
say(f"lambda_g = lambda/n_eff = {LAM/neff_ref*1e3:.1f} nm; evanescent decay length 1/gamma = {gamma_ref*1e3:.1f} nm")

# --- tail decay length measured from the solver field along +x (outside the sidewall)
def tail_decay_length(fields):
    iy = np.argmin(np.abs(YS))
    prof = np.abs(fields["Ex"][:, iy]) ** 2
    sel = (XS >= 0.40) & (XS <= 0.75)                    # well outside the 250 nm half-width
    slope = np.polyfit(XS[sel], np.log(prof[sel]), 1)[0]  # |E|^2 ~ e^{-2 x / delta}
    return -2.0 / slope

tail_td, tail_fw = tail_decay_length(te0_td.fields), tail_decay_length(te0_fw.fields)
say(f"|E|^2 tail fit along x: Tidy3D {tail_td*1e3:.1f} nm, femwell {tail_fw*1e3:.1f} nm vs 1/gamma {gamma_ref*1e3:.1f} nm")

# --- E_x jump at the sidewall (normal component: D_x continuous -> E_x jumps by eps ratio)
def sidewall_jump(fields):
    iy = np.argmin(np.abs(YS))
    xin, xout = np.argmin(np.abs(XS - 0.24)), np.argmin(np.abs(XS - 0.26))
    return abs(fields["Ex"][xout, iy]) / abs(fields["Ex"][xin, iy])

jump_td, jump_fw = sidewall_jump(te0_td.fields), sidewall_jump(te0_fw.fields)
ez_over_ex_td = np.abs(te0_td.fields["Ez"]).max() / np.abs(te0_td.fields["Ex"]).max()
ez_over_ex_fw = np.abs(te0_fw.fields["Ez"]).max() / np.abs(te0_fw.fields["Ex"]).max()
say(f"peak |E_z| / peak |E_x| of TE0: Tidy3D {ez_over_ex_td:.3f}, femwell {ez_over_ex_fw:.3f}")
cx, cex = te0_fw.fields["Ex_cut"]
jump_wall = cex[4] / cex[3]
say(f"E_x outside/inside the sidewall, 10 nm either side (grid-limited): Tidy3D {jump_td:.2f}, femwell {jump_fw:.2f}; "
    f"femwell 0.1 nm either side: {jump_wall:.2f}; eps ratio (n_Si/n_SiO2)^2 = {(NSI/NSIO2)**2:.2f} "
    f"({pct(jump_wall, (NSI/NSIO2)**2):+.1f} %)")

# ----------------------------------------------------------------- figure: TE0 fields, both solvers
def core_box(ax, w=W, h=H):
    ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fill=False, ec=PALETTE["ink"], lw=1.0, ls="--"))


extent = (XS[0], XS[-1], YS[0], YS[-1])
fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
for row, m in enumerate((te0_td, te0_fw)):
    F = m.fields
    E2 = sum(np.abs(F[c]) ** 2 for c in ("Ex", "Ey", "Ez"))
    panels = [("Ex", np.real(F["Ex"]), "RdBu_r", "E_x (horizontal, TE component)"),
              ("Ey", np.real(F["Ey"]), "RdBu_r", "E_y (vertical, small)"),
              ("Ez", np.imag(F["Ez"]), "RdBu_r", "Im E_z (longitudinal, 90 deg off)"),
              ("E2", E2, "Blues", "|E|^2 (intensity)")]
    for col, (key, arr, cmap, ttl) in enumerate(panels):
        ax = axes[row, col]
        if cmap == "RdBu_r":
            vmax = np.abs(arr).max()
            im = ax.imshow(arr.T, origin="lower", extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        else:
            im = ax.imshow(arr.T, origin="lower", extent=extent, cmap=cmap, vmin=0, aspect="equal")
        core_box(ax)
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("x, width direction (um)"); ax.set_ylabel("y, height direction (um)")
        ax.set_title(f"{m.solver}: {ttl}", fontsize=9)
        ax.grid(False)
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("field (a.u., solver normalisation)" if cmap == "RdBu_r" else "|E|^2 (a.u., solver normalisation)", fontsize=8)
fig.suptitle(f"TE0 mode of the {W*1e3:.0f}x{H*1e3:.0f} nm strip at {LAM*1e3:.0f} nm: both solvers give the same shape, "
             f"n_eff = {te0_td.n_eff:.3f} (Tidy3D) / {te0_fw.n_eff:.3f} (femwell); dashed = silicon core", fontsize=11)
fig.savefig(OUT / "08_mode_fields.png")
plt.close(fig)

# ----------------------------------------------------------------- figure: mode gallery (all guided modes)
labels = [m.label for m in fw_ref]
fig, axes = plt.subplots(2, len(labels), figsize=(3.6 * len(labels), 6.0), constrained_layout=True)
for col, lab in enumerate(labels):
    for row, ms in enumerate((td_ref, fw_ref)):
        m = find_mode(ms, lab)
        ax = axes[row, col]
        if m is None:
            ax.set_axis_off(); continue
        F = m.fields
        E2 = sum(np.abs(F[c]) ** 2 for c in ("Ex", "Ey", "Ez"))
        ax.imshow((E2 / E2.max()).T, origin="lower", extent=extent, cmap="Blues", vmin=0, aspect="equal")
        core_box(ax)
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.55, 0.55); ax.grid(False)
        ax.set_title(f"{m.solver} {lab}: n_eff={m.n_eff:.3f}\nTE={m.te_fraction:.2f}, Gamma_P={m.gamma_P:.2f}", fontsize=9)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
fig.suptitle(f"All guided modes (n_eff > {NSIO2}) of the 500x220 nm strip at 1310 nm: it is NOT single-mode, "
             "TE1 is guided (weakly)", fontsize=11)
fig.savefig(OUT / "08_mode_gallery.png")
plt.close(fig)

# ----------------------------------------------------------------- figure: profile cuts + tails
iy0, ix0 = np.argmin(np.abs(YS)), np.argmin(np.abs(XS))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
for m, c in ((te0_td, SERIES[0]), (te0_fw, SERIES[1])):
    ex = np.real(m.fields["Ex"])
    axes[0].plot(XS * 1e3, ex[:, iy0] / ex[ix0, iy0], color=c, label=f"{m.solver}")
    axes[1].plot(YS * 1e3, ex[ix0, :] / ex[ix0, iy0], color=c, label=f"{m.solver}")
    prof = np.abs(ex[:, iy0]) ** 2
    axes[2].semilogy(XS * 1e3, prof / prof.max(), color=c, label=f"{m.solver} |E_x|^2")
xx = np.linspace(0.25, 1.0, 50)
axes[2].semilogy(xx * 1e3, np.exp(-2 * (xx - 0.25) / gamma_ref) * 0.08, "k--", lw=1.2,
                 label=f"e^(-2x/delta), delta = 1/gamma = {gamma_ref*1e3:.0f} nm (notes 16)")
for ax, lab in ((axes[0], "x"), (axes[2], "x")):
    for s in (-1, 1):
        ax.axvline(s * W / 2 * 1e3, color=PALETTE["muted"], lw=0.8, ls=":")
for s in (-1, 1):
    axes[1].axvline(s * H / 2 * 1e3, color=PALETTE["muted"], lw=0.8, ls=":")
axes[0].set_title("E_x across the width jumps at the sidewalls (D_x is continuous, E_x is not)", fontsize=9.5)
axes[0].set_xlabel("x (nm)"); axes[0].set_ylabel("E_x / E_x(0,0)")
axes[0].plot(cx * 1e3, cex / cex[0], "k.", ms=4, label="femwell probed 0.1 nm from the wall")
axes[0].annotate(f"jump at the wall = {jump_wall:.2f}\n(eps ratio {(NSI/NSIO2)**2:.2f})", xy=(270, 0.55), fontsize=8)
axes[1].set_title("E_x across the height is continuous (tangential to the top/bottom faces)", fontsize=9.5)
axes[1].set_xlabel("y (nm)"); axes[1].set_ylabel("E_x / E_x(0,0)")
axes[2].set_title(f"Evanescent tail: fitted decay {tail_fw*1e3:.0f} nm (femwell), {tail_td*1e3:.0f} nm (Tidy3D)", fontsize=9.5)
axes[2].set_xlabel("x (nm)"); axes[2].set_ylabel("|E_x|^2 / max"); axes[2].set_ylim(1e-6, 1.5); axes[2].set_xlim(-1000, 1000)
axes[0].legend(fontsize=8, loc="upper left"); axes[1].legend(fontsize=8); axes[2].legend(fontsize=8, loc="upper right")
fig.savefig(OUT / "08_profile_cuts.png")
plt.close(fig)

# ================================================================== B. group index decomposition
say("\n--- B. group index: waveguide dispersion vs material dispersion")
t = time.time()
ng_wg_td = te0_td.n_g_wg                          # Tidy3D finite-difference in frequency, fixed materials

def fw_neff(lam, nsi=NSI, nsio2=NSIO2):
    return solve_femwell(fm_ref, lam, nsi, nsio2, num_modes=1)[0].n_eff

DLAM = 0.005
ng_wg_fw = te0_fw.n_eff - LAM * (fw_neff(LAM + DLAM) - fw_neff(LAM - DLAM)) / (2 * DLAM)

# brute force with Sellmeier materials: n_Si(lam), n_SiO2(lam) move with lam
def td_neff_sellmeier(lam):
    return solve_tidy3d(W, H, lam, float(n_si(lam)), float(n_sio2(lam)), num_modes=1, dl=TD_DL,
                        box=TD_BOX, group_index=False)[0].n_eff

n_td_p, n_td_m = td_neff_sellmeier(LAM + DLAM), td_neff_sellmeier(LAM - DLAM)
n_td_0 = solve_tidy3d(W, H, LAM, float(n_si(LAM)), float(n_sio2(LAM)), num_modes=1, dl=TD_DL, box=TD_BOX,
                      group_index=False)[0].n_eff
ng_tot_td = n_td_0 - LAM * (n_td_p - n_td_m) / (2 * DLAM)
n_fw_0 = fw_neff(LAM, float(n_si(LAM)), float(n_sio2(LAM)))
ng_tot_fw = n_fw_0 - LAM * (fw_neff(LAM + DLAM, float(n_si(LAM + DLAM)), float(n_sio2(LAM + DLAM)))
                            - fw_neff(LAM - DLAM, float(n_si(LAM - DLAM)), float(n_sio2(LAM - DLAM)))) / (2 * DLAM)

ng_si_bulk, ng_sio2_bulk = float(group_index(n_si, LAM)), float(group_index(n_sio2, LAM))
mat_term_fw = te0_fw.S_core * (ng_si_bulk - float(n_si(LAM))) + te0_fw.S_clad * (ng_sio2_bulk - float(n_sio2(LAM)))
mat_term_td = te0_td.S_core * (ng_si_bulk - float(n_si(LAM))) + te0_td.S_clad * (ng_sio2_bulk - float(n_sio2(LAM)))
ng_pred_fw, ng_pred_td = ng_wg_fw + mat_term_fw, ng_wg_td + mat_term_td
say(f"bulk: n_Si={float(n_si(LAM)):.4f}, n_g,Si={ng_si_bulk:.4f}; n_SiO2={float(n_sio2(LAM)):.4f}, n_g,SiO2={ng_sio2_bulk:.4f}")
say(f"n_g (fixed materials = waveguide dispersion only): Tidy3D {ng_wg_td:.4f}, femwell {ng_wg_fw:.4f}")
say(f"n_g (Sellmeier materials, brute-force d/dlambda): Tidy3D {ng_tot_td:.4f}, femwell {ng_tot_fw:.4f}")
say(f"n_g predicted = n_g,wg + S_Si (n_g,Si - n_Si) + S_SiO2 (n_g,SiO2 - n_SiO2): Tidy3D {ng_pred_td:.4f}, femwell {ng_pred_fw:.4f}")
say(f"   waveguide-dispersion part n_g,wg - n_eff = {ng_wg_fw - te0_fw.n_eff:+.3f}, material part = {mat_term_fw:+.3f}")
ng_solver = 0.5 * (ng_tot_td + ng_tot_fw)
say(f"REF n_g = {REF.ng}: solver mean {ng_solver:.3f} ({pct(ng_solver, REF.ng):+.1f} %)")
fsr_nm_solver = LAM**2 / (ng_solver * L_RT) * 1e3
fsr_thz_solver = C0 / (ng_solver * L_RT * 1e-6) / 1e12
say(f"FSR = lambda^2/(n_g L): {fsr_nm_solver:.2f} nm / {fsr_thz_solver:.3f} THz with solver n_g (REF {REF.fsr_nm} nm / {REF.fsr_thz} THz)")
say(f"(B took {time.time()-t:.1f} s)")

# ================================================================== C. width sweep
say("\n--- C. width sweep 300-700 nm (fixed REF indices)")
t = time.time()
widths = np.round(np.arange(0.30, 0.7001, 0.02), 3)
sw = {k: [] for k in ("w", "td_TE0", "td_TM0", "td_TE1", "fw_TE0", "fw_TM0", "fw_TE1",
                      "td_ng", "fw_ng", "td_gP", "fw_gP", "td_S", "fw_S", "fw_Sclad", "fw_TE0_te", "td_TE0_te")}
frames = []      # femwell TE0 fields for the video
for w in widths:
    td = solve_tidy3d(w, H, LAM, NSI, NSIO2, num_modes=3, dl=TD_DL, box=TD_BOX)
    fm = femwell_mesh(w, H, resolution=FEM_RES_SWEEP, box=2.0)
    fw = solve_femwell(fm, LAM, NSI, NSIO2, num_modes=3, want_fields=True)
    fw_p = solve_femwell(fm, LAM + DLAM, NSI, NSIO2, num_modes=1)[0].n_eff
    fw_m = solve_femwell(fm, LAM - DLAM, NSI, NSIO2, num_modes=1)[0].n_eff
    a, b = find_mode(td, "TE0"), find_mode(fw, "TE0")
    sw["w"].append(w)
    for key, ms in (("td", td), ("fw", fw)):
        for lab in ("TE0", "TM0", "TE1"):
            m = find_mode(ms, lab)
            sw[f"{key}_{lab}"].append(m.n_eff if m else np.nan)
    sw["td_ng"].append(a.n_g_wg); sw["fw_ng"].append(b.n_eff - LAM * (fw_p - fw_m) / (2 * DLAM))
    sw["td_gP"].append(a.gamma_P); sw["fw_gP"].append(b.gamma_P)
    sw["td_S"].append(a.S_core); sw["fw_S"].append(b.S_core); sw["fw_Sclad"].append(b.S_clad)
    sw["td_TE0_te"].append(a.te_fraction); sw["fw_TE0_te"].append(b.te_fraction)
    frames.append((w, b.n_eff, b.gamma_P, np.real(b.fields["Ex"]),
                   sum(np.abs(b.fields[c]) ** 2 for c in ("Ex", "Ey", "Ez"))))
    say(f"  w={w*1e3:4.0f} nm: TE0 {a.n_eff:.4f}/{b.n_eff:.4f}  TM0 {sw['td_TM0'][-1]:.4f}/{sw['fw_TM0'][-1]:.4f}  "
        f"TE1 {sw['td_TE1'][-1]:.4f}/{sw['fw_TE1'][-1]:.4f}  n_g,wg {a.n_g_wg:.3f}/{sw['fw_ng'][-1]:.3f}  Gamma_P {a.gamma_P:.3f}/{b.gamma_P:.3f}")
sw = {k: np.array(v, dtype=float) for k, v in sw.items()}
sw["fw_ng_total"] = sw["fw_ng"] + sw["fw_S"] * (ng_si_bulk - NSI) + sw["fw_Sclad"] * (ng_sio2_bulk - NSIO2)
# widths at which the femwell TE0 n_eff crosses the textbook 2.5 and the total n_g crosses the textbook 4.2
neff_2p5_w = float(np.interp(REF.neff, sw["fw_TE0"], sw["w"])) * 1e3                       # n_eff rises with w
ng_4p2_w = float(np.interp(REF.ng, sw["fw_ng_total"][::-1], sw["w"][::-1])) * 1e3            # n_g falls with w
say(f"femwell TE0 n_eff crosses the textbook {REF.neff} at w = {neff_2p5_w:.0f} nm; total n_g crosses {REF.ng} at w = {ng_4p2_w:.0f} nm")

def cutoff_width(wv, neff_te1, n_clad=NSIO2, margin=0.003):
    """Width at which TE1 reaches n_clad + margin (margin: box-truncated near-cutoff modes are unreliable)."""
    ok = np.isfinite(neff_te1) & (neff_te1 > n_clad + margin)
    if ok.sum() < 2:
        return np.nan
    i = np.argmax(ok)                      # first guided point
    if i == 0:
        return np.nan
    x0, x1 = wv[i - 1], wv[i]
    y0 = neff_te1[i - 1] if np.isfinite(neff_te1[i - 1]) else n_clad
    y1 = neff_te1[i]
    return x0 + (n_clad + margin - y0) * (x1 - x0) / (y1 - y0)

cut_td, cut_fw = cutoff_width(sw["w"], sw["td_TE1"]), cutoff_width(sw["w"], sw["fw_TE1"])
say(f"TE1 cutoff width (n_eff = n_SiO2 + 0.003): Tidy3D {cut_td*1e3:.0f} nm, femwell {cut_fw*1e3:.0f} nm")
i500 = np.argmin(np.abs(sw["w"] - 0.5))
dneff_dw_td = (sw["td_TE0"][i500 + 1] - sw["td_TE0"][i500 - 1]) / (0.04 * 1e3)     # per nm
dneff_dw_fw = (sw["fw_TE0"][i500 + 1] - sw["fw_TE0"][i500 - 1]) / (0.04 * 1e3)
dlam_dw_fw = LAM * 1e3 * dneff_dw_fw / ng_solver     # nm of resonance shift per nm of width
say(f"dn_eff/dw at 500 nm: Tidy3D {dneff_dw_td:.2e}/nm, femwell {dneff_dw_fw:.2e}/nm -> "
    f"dlambda_r/dw = lambda (dn_eff/dw)/n_g = {dlam_dw_fw*1e3:.0f} pm per nm of width; "
    f"10 nm width error = {dlam_dw_fw*10:.2f} nm = {dlam_dw_fw*10/fsr_nm_solver:.2f} FSR")
say(f"(C took {time.time()-t:.1f} s)")

dldT_guess = LAM * 1e6 * te0_fw.S_core * REF.dn_si_dT / ng_solver     # pm/K, refined in section E
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
ax = axes[0, 0]
for lab, c in (("TE0", SERIES[0]), ("TM0", SERIES[1]), ("TE1", SERIES[2])):
    ax.plot(sw["w"] * 1e3, sw[f"td_{lab}"], "o-", color=c, ms=3, label=f"{lab} Tidy3D")
    ax.plot(sw["w"] * 1e3, sw[f"fw_{lab}"], "s--", color=c, ms=3, mfc="none", label=f"{lab} femwell")
ax.axhline(NSIO2, color=PALETTE["muted"], ls=":", lw=1); ax.text(560, NSIO2 + 0.02, "n_SiO2 = 1.45 (cutoff line)", fontsize=8)
ax.axhline(REF.neff, color=PALETTE["red"], ls=":", lw=1); ax.text(560, REF.neff + 0.02, "textbook REF n_eff = 2.5", fontsize=8, color=PALETTE["red"])
if np.isfinite(cut_fw):
    ax.axvline(cut_fw * 1e3, color=SERIES[2], ls=":", lw=1); ax.text(cut_fw * 1e3 + 4, 1.62, f"TE1 cutoff\n~{cut_fw*1e3:.0f} nm", fontsize=8, color=SERIES[2])
ax.axvline(500, color=PALETTE["line"], lw=6, alpha=0.6, zorder=0)
ax.set_xlabel("strip width (nm)"); ax.set_ylabel("n_eff"); ax.legend(fontsize=7, ncol=2)
ax.set_title(f"n_eff vs width: TE0 rises toward n_Si; TE1 guided above ~{cut_fw*1e3:.0f} nm, so 500 nm is bimodal", fontsize=9.5)
ax = axes[0, 1]
ax.plot(sw["w"] * 1e3, sw["td_ng"], "o-", color=SERIES[0], ms=3, label="n_g,wg Tidy3D (fixed n)")
ax.plot(sw["w"] * 1e3, sw["fw_ng"], "s--", color=SERIES[1], ms=3, mfc="none", label="n_g,wg femwell (fixed n)")
ax.plot(sw["w"] * 1e3, sw["fw_ng_total"], ":", color=SERIES[3], label="n_g total femwell (+ material dispersion of Si and SiO2)")
ax.axhline(REF.ng, color=PALETTE["red"], ls=":", lw=1); ax.text(302, REF.ng + 0.02, "REF n_g = 4.2", fontsize=8, color=PALETTE["red"])
ax.set_xlabel("strip width (nm)"); ax.set_ylabel("group index"); ax.legend(fontsize=8)
ax.set_title("Group index of TE0: waveguide dispersion falls with width, material dispersion adds ~0.2", fontsize=9.5)
ax = axes[1, 0]
ax.plot(sw["w"] * 1e3, sw["td_gP"], "o-", color=SERIES[0], ms=3, label="Gamma_P (power in Si) Tidy3D")
ax.plot(sw["w"] * 1e3, sw["fw_gP"], "s--", color=SERIES[1], ms=3, mfc="none", label="Gamma_P femwell")
ax.plot(sw["w"] * 1e3, sw["td_S"], "o-", color=SERIES[2], ms=3, label="S_Si = dn_eff/dn_Si Tidy3D")
ax.plot(sw["w"] * 1e3, sw["fw_S"], "s--", color=SERIES[3], ms=3, mfc="none", label="S_Si femwell")
ax.axhline(REF.confinement, color=PALETTE["red"], ls=":", lw=1); ax.text(302, REF.confinement - 0.05, "REF Gamma = 0.85", fontsize=8, color=PALETTE["red"])
ax.set_xlabel("strip width (nm)"); ax.set_ylabel("fraction / sensitivity"); ax.legend(fontsize=8)
ax.set_title("Two different 'confinements': power fraction Gamma_P < 1, but dn_eff/dn_Si > 1", fontsize=9.5)
ax = axes[1, 1]
dndw = np.gradient(sw["fw_TE0"], sw["w"] * 1e3)
ax.plot(sw["w"] * 1e3, LAM * 1e3 * dndw / ng_solver * 1e3, "s-", color=SERIES[1], ms=3, label="femwell")
dndw_td = np.gradient(sw["td_TE0"], sw["w"] * 1e3)
ax.plot(sw["w"] * 1e3, LAM * 1e3 * dndw_td / ng_solver * 1e3, "o--", color=SERIES[0], ms=3, mfc="none", label="Tidy3D")
ax.set_xlabel("strip width (nm)"); ax.set_ylabel("resonance shift per nm of width (pm/nm)"); ax.legend(fontsize=8)
ax.set_title(f"Fabrication: at 500 nm, 1 nm of width moves lambda_r by {dlam_dw_fw*1e3:.0f} pm "
             f"(~{dlam_dw_fw*1e3/dldT_guess:.0f} K of heating)", fontsize=9.5)
fig.suptitle("Width sweep of the 220 nm-thick SOI strip at 1310 nm (both solvers, fixed n_Si = 3.50, n_SiO2 = 1.45)")
fig.savefig(OUT / "08_width_sweep.png")
plt.close(fig)

# ================================================================== D. wavelength sweep
say("\n--- D. wavelength sweep 1260-1360 nm (Sellmeier n_Si(lambda), n_SiO2(lambda))")
t = time.time()
lams = np.round(np.arange(1.26, 1.3601, 0.01), 4)
wl = {k: [] for k in ("lam", "td_neff", "fw_neff", "td_ng_wg", "td_ng_tot", "td_gP", "fw_gP", "fw_S", "td_S")}
for lam in lams:
    nsi_l, nsio2_l = float(n_si(lam)), float(n_sio2(lam))
    a = solve_tidy3d(W, H, lam, nsi_l, nsio2_l, num_modes=1, dl=TD_DL, box=TD_BOX)[0]
    b = solve_femwell(fm_ref, lam, nsi_l, nsio2_l, num_modes=1)[0]
    ng_tot = a.n_g_wg + a.S_core * (float(group_index(n_si, lam)) - nsi_l) + a.S_clad * (float(group_index(n_sio2, lam)) - nsio2_l)
    wl["lam"].append(lam); wl["td_neff"].append(a.n_eff); wl["fw_neff"].append(b.n_eff)
    wl["td_ng_wg"].append(a.n_g_wg); wl["td_ng_tot"].append(ng_tot)
    wl["td_gP"].append(a.gamma_P); wl["fw_gP"].append(b.gamma_P); wl["fw_S"].append(b.S_core); wl["td_S"].append(a.S_core)
    say(f"  lam={lam*1e3:.0f} nm: n_eff {a.n_eff:.4f}/{b.n_eff:.4f}  n_g,wg {a.n_g_wg:.3f}  n_g,total {ng_tot:.3f}  Gamma_P {a.gamma_P:.3f}/{b.gamma_P:.3f}")
wl = {k: np.array(v) for k, v in wl.items()}
wl["fw_ng_tot"] = wl["fw_neff"] - wl["lam"] * np.gradient(wl["fw_neff"], wl["lam"])    # total n_g from the sweep itself
say(f"femwell n_g,total from the sweep's own d n_eff/d lambda at 1310: {wl['fw_ng_tot'][5]:.4f} (brute force above: {ng_tot_fw:.4f})")
say(f"(D took {time.time()-t:.1f} s)")

f0 = C0 / (LAM * 1e-6)
chan_f = f0 + (np.arange(REF.n_channels) - (REF.n_channels - 1) / 2) * REF.channel_spacing_ghz * 1e9
chan_lam = C0 / chan_f * 1e6
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
ax = axes[0]
ax.plot(wl["lam"] * 1e3, wl["td_neff"], "o-", ms=3, label="Tidy3D"); ax.plot(wl["lam"] * 1e3, wl["fw_neff"], "s--", ms=3, mfc="none", label="femwell")
ax.axhline(REF.neff, color=PALETTE["red"], ls=":", lw=1); ax.text(1262, REF.neff + 0.01, "REF n_eff = 2.5", fontsize=8, color=PALETTE["red"])
for cl in chan_lam:
    ax.axvline(cl * 1e3, color=PALETTE["line"], lw=1, alpha=0.3, zorder=0)
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("n_eff (TE0)"); ax.legend(fontsize=8)
ax.set_title("n_eff falls with wavelength; grey lines = the 8 channels, 200 GHz apart", fontsize=9)
ax = axes[1]
ax.plot(wl["lam"] * 1e3, wl["td_ng_tot"], "o-", ms=3, label="n_g total, Tidy3D (decomposition)")
ax.plot(wl["lam"] * 1e3, wl["fw_ng_tot"], "s--", ms=3, mfc="none", label="n_g total, femwell (d n_eff/d lambda of sweep)")
ax.plot(wl["lam"] * 1e3, wl["td_ng_wg"], ":", label="n_g waveguide-only, Tidy3D")
ax.axhline(REF.ng, color=PALETTE["red"], ls=":", lw=1); ax.text(1262, REF.ng + 0.01, "REF n_g = 4.2", fontsize=8, color=PALETTE["red"])
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("group index (TE0)"); ax.legend(fontsize=8)
ax.set_title("Group index across the O-band: the FSR-setting number", fontsize=9)
ax = axes[2]
ax.plot(wl["lam"] * 1e3, wl["td_gP"], "o-", ms=3, label="Gamma_P Tidy3D"); ax.plot(wl["lam"] * 1e3, wl["fw_gP"], "s--", ms=3, mfc="none", label="Gamma_P femwell")
ax.plot(wl["lam"] * 1e3, wl["td_S"], "o-", ms=3, label="S_Si Tidy3D"); ax.plot(wl["lam"] * 1e3, wl["fw_S"], "s--", ms=3, mfc="none", label="S_Si femwell")
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("fraction / sensitivity"); ax.legend(fontsize=8)
ax.set_title("Confinement is nearly flat: one dlambda/dT serves all 8 channels", fontsize=9)
fig.suptitle("Wavelength sweep of the 500x220 nm strip with Sellmeier silicon and silica")
fig.savefig(OUT / "08_wavelength_sweep.png")
plt.close(fig)

# ================================================================== E. thermo-optic shift
say("\n--- E. thermo-optic perturbation (the capstone number)")
t = time.time()
DT = 10.0
dnsi, dnsio2 = REF.dn_si_dT * DT, REF.dn_sio2_dT * DT
res_to = {}
for key, solver in (("tidy3d", lambda a, b: solve_tidy3d(W, H, LAM, a, b, num_modes=1, dl=TD_DL, box=TD_BOX, group_index=False)[0].n_eff),
                    ("femwell", lambda a, b: fw_neff(LAM, a, b))):
    n0 = solver(NSI, NSIO2)
    n_si_only = solver(NSI + dnsi, NSIO2)
    n_sio2_only = solver(NSI, NSIO2 + dnsio2)
    n_both = solver(NSI + dnsi, NSIO2 + dnsio2)
    res_to[key] = dict(n0=n0, S_si_direct=(n_si_only - n0) / dnsi, S_sio2_direct=(n_sio2_only - n0) / dnsio2,
                       dneff_dT=(n_both - n0) / DT)
    say(f"  {key}: n_eff(0)={n0:.6f}, +{DT:.0f} K -> {n_both:.6f}; dn_eff/dT = {res_to[key]['dneff_dT']:.4e} /K; "
        f"direct S_Si = {res_to[key]['S_si_direct']:.4f}, S_SiO2 = {res_to[key]['S_sio2_direct']:.4f}")

# linearity over the whole ambient range (femwell)
dTs = np.array([0, 10, 25, 50, 75, 100, 115.0])
neff_T = np.array([fw_neff(LAM, NSI + REF.dn_si_dT * d, NSIO2 + REF.dn_sio2_dT * d) for d in dTs])
lin = np.polyfit(dTs, neff_T, 1)
say(f"  femwell n_eff(dT) over 0-115 K: slope {lin[0]:.4e} /K, max deviation from linear {np.abs(neff_T - np.polyval(lin, dTs)).max():.1e}")

# resonance shift: d lambda_r / dT = lambda * (dn_eff/dT) / n_g  (total group index, see README)
textbook_est = LAM * 1e6 * REF.confinement * REF.dn_si_dT / REF.ng      # pm/K, the Gamma-weighted estimate
S_mean = 0.5 * (res_to["tidy3d"]["S_si_direct"] + res_to["femwell"]["S_si_direct"])
pert_est = LAM * 1e6 * (S_mean * REF.dn_si_dT + 0.5 * (te0_td.S_clad + te0_fw.S_clad) * REF.dn_sio2_dT) / REF.ng
shift = {}
for key in ("tidy3d", "femwell"):
    ng_use = ng_tot_td if key == "tidy3d" else ng_tot_fw
    shift[key] = dict(ng_solver=LAM * 1e6 * res_to[key]["dneff_dT"] / ng_use,
                      ng_ref=LAM * 1e6 * res_to[key]["dneff_dT"] / REF.ng)
say(f"  d lambda_r/dT [pm/K]: textbook Gamma-weighted (0.85 x 1.86e-4, n_g 4.2) = {textbook_est:.1f}; REF = {REF.dlambda_dT_pm_per_K}")
for key in ("tidy3d", "femwell"):
    say(f"     {key}: {shift[key]['ng_solver']:.1f} pm/K with solver n_g, {shift[key]['ng_ref']:.1f} pm/K with REF n_g 4.2")
dldT = 0.5 * (shift["tidy3d"]["ng_solver"] + shift["femwell"]["ng_solver"])
say(f"  solver mean {dldT:.1f} pm/K = {pct(dldT, REF.dlambda_dT_pm_per_K):+.0f} % vs REF 50 pm/K; "
    f"perturbation formula (S_Si dn_Si/dT + S_SiO2 dn_SiO2/dT) lambda/n_g,REF = {pert_est:.1f} pm/K")
# thermal expansion: m lambda_r = n_eff(lambda_r, T) L(T) at fixed m gives
#   d lambda_r/dT = (lambda/n_g) (dn_eff/dT + n_eff alpha_L)   with alpha_L = dL/dT / L of silicon
ALPHA_L_SI = 2.6e-6                                             # /K, linear expansion of crystalline Si near 300 K
dneff_dT_mean = 0.5 * (res_to["tidy3d"]["dneff_dT"] + res_to["femwell"]["dneff_dT"])
exp_term_index = neff_ref * ALPHA_L_SI                          # /K, the n_eff alpha_L term in dn_eff/dT units
exp_term_pm = LAM * 1e6 * exp_term_index / ng_solver            # pm/K
exp_frac = exp_term_index / dneff_dT_mean
geom_term_index = dneff_dw_fw * (W * 1e3) * ALPHA_L_SI          # /K: the cross-section also widens, dn_eff/dw * w * alpha_L
dldT_total = dldT + exp_term_pm
say(f"  thermal expansion of the ring (alpha_L = {ALPHA_L_SI:.1e}/K): n_eff alpha_L = {exp_term_index:.2e}/K = {exp_frac*100:.1f} % of dn_eff/dT "
    f"-> +{exp_term_pm:.1f} pm/K; total d lambda_r/dT = {dldT_total:.1f} pm/K ({pct(dldT_total, REF.dlambda_dT_pm_per_K):+.0f} % vs REF); "
    f"the widening of the cross-section adds only {geom_term_index:.1e}/K ({geom_term_index/dneff_dT_mean*100:.1f} %), ignored")
one_K_frac = dldT / REF.fwhm_pm
say(f"  1 K = {dldT:.0f} pm = {one_K_frac*100:.0f} % of the {REF.fwhm_pm:.0f} pm FWHM (REF: {REF.dlambda_dT_pm_per_K/REF.fwhm_pm*100:.0f} %); "
    f"0.1 K = {dldT/10:.1f} pm; delta_opt = {REF.delta_opt_pm} pm = {REF.delta_opt_pm/dldT:.1f} K")
amb = REF.ambient_max_c - REF.ambient_min_c
say(f"  ambient span {amb:.0f} K -> {amb*dldT*1e-3:.2f} nm of drift = {amb*dldT*1e-3/fsr_nm_solver:.2f} FSR "
    f"(REF: {amb*REF.dlambda_dT_pm_per_K*1e-3:.2f} nm = {amb*REF.dlambda_dT_pm_per_K*1e-3/REF.fsr_nm:.2f} FSR); "
    f"channel spacing {abs(chan_lam[1]-chan_lam[0])*1e3:.2f} nm = {abs(chan_lam[1]-chan_lam[0])*1e6/dldT:.0f} K of drift; one FSR = {fsr_nm_solver*1e3/dldT:.0f} K")
say(f"(E took {time.time()-t:.1f} s)")

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
ax = axes[0, 0]
ax.plot(dTs, (neff_T - neff_T[0]) * 1e3, "s-", color=SERIES[1], label="femwell, n_Si and n_SiO2 both perturbed")
ax.plot([DT], [(res_to["tidy3d"]["dneff_dT"] * DT) * 1e3], "o", color=SERIES[0], ms=7, label="Tidy3D at +10 K")
ax.plot(dTs, REF.confinement * REF.dn_si_dT * dTs * 1e3, ":", color=PALETTE["red"], label="textbook: Gamma_P x dn_Si/dT (0.85 x 1.86e-4)")
ax.set_xlabel("temperature rise dT (K)"); ax.set_ylabel("dn_eff  (x 1e-3)"); ax.legend(fontsize=8)
ax.set_title(f"n_eff rises linearly with T: slope {lin[0]:.3e}/K, i.e. {lin[0]/REF.dn_si_dT:.2f} x dn_Si/dT (not 0.85 x)", fontsize=9.5)
ax = axes[0, 1]
names = ["REF\n(50)", "textbook\nGamma_P\nx dn/dT", "S_Si\nformula\nn_g 4.2", "Tidy3D\nsolver\nn_g", "femwell\nsolver\nn_g", "Tidy3D\nn_g 4.2", "femwell\nn_g 4.2"]
vals = [REF.dlambda_dT_pm_per_K, textbook_est, pert_est, shift["tidy3d"]["ng_solver"], shift["femwell"]["ng_solver"], shift["tidy3d"]["ng_ref"], shift["femwell"]["ng_ref"]]
cols = [PALETTE["red"], PALETTE["red"], SERIES[2], SERIES[0], SERIES[1], SERIES[0], SERIES[1]]
bars = ax.bar(range(len(vals)), vals, color=cols)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylabel("d lambda_r / dT (pm/K)")
ax.set_title("Resonance shift per kelvin: solvers ~63, textbook Gamma_P-weighting ~49", fontsize=9.5)
ax = axes[1, 0]
dl = np.linspace(-800, 800, 1601)
def lorentz(d, t_min=REF.t_min, fwhm=REF.fwhm_pm):
    return 1 - (1 - t_min) / (1 + (2 * d / fwhm) ** 2)
ax.plot(dl, lorentz(dl), color=SERIES[0], label="ring through-port at T0 (FWHM 374 pm)")
ax.plot(dl, lorentz(dl - dldT), color=SERIES[1], label=f"after +1 K: shifted {dldT:.0f} pm")
ax.plot(dl, lorentz(dl - REF.delta_opt_pm), ":", color=SERIES[3], label=f"after +{REF.delta_opt_pm/dldT:.1f} K: shifted by delta_opt = 108 pm")
ax.axvline(REF.delta_opt_pm, color=PALETTE["ink2"], lw=1, ls="--"); ax.text(115, 0.05, "laser at delta_opt = +108 pm", fontsize=8)
ax.set_xlabel("laser wavelength relative to the cold resonance, lambda_L - lambda_r(T0) (pm)"); ax.set_ylabel("transmission"); ax.legend(fontsize=8)
ax.set_title(f"What 1 K does to the notch the laser sits on: {one_K_frac*100:.0f} % of a linewidth", fontsize=9.5)
ax = axes[1, 1]
Tamb = np.linspace(REF.ambient_min_c, REF.ambient_max_c, 200)
lam_r = LAM * 1e3 + (Tamb - 25) * dldT * 1e-3
for k in range(-2, 3):
    ax.plot(Tamb, lam_r + k * fsr_nm_solver, color=SERIES[0], lw=1.5 if k == 0 else 0.8, label="ring resonances (solver dlambda/dT), one per FSR" if k == 0 else None)
ax.plot(Tamb, LAM * 1e3 + (Tamb - 25) * REF.dlambda_dT_pm_per_K * 1e-3, "--", color=PALETTE["red"], lw=1, label="same with REF 50 pm/K")
for i, cl in enumerate(chan_lam):
    ax.axhline(cl * 1e3, color=PALETTE["muted"], lw=0.8, ls=":", label="8 laser channels, 200 GHz apart" if i == 0 else None)
ax.set_ylim((lam_r - 2 * fsr_nm_solver).min() - 1.5, (lam_r + 2 * fsr_nm_solver).max() + 11)   # headroom for the legend
ax.set_xlabel("ambient temperature (C)"); ax.set_ylabel("wavelength (nm)"); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"Free-running ring, 10-125 C: drifts {amb*dldT*1e-3:.1f} nm, crosses a channel every ~{abs(chan_lam[1]-chan_lam[0])*1e6/dldT:.0f} K", fontsize=9.5)
fig.suptitle("Thermo-optic shift of the 500x220 nm strip: perturb n_Si by 1.86e-4/K and n_SiO2 by 1e-5/K, re-solve the mode")
fig.savefig(OUT / "08_thermo_optic.png")
plt.close(fig)

# ================================================================== F. n_eff vs n_g: ring comb
say("\n--- F. which ring numbers depend on n_eff and which on n_g")
lam_axis = np.linspace(1.290, 1.330, 20001)
def allpass(lam, neff0, ng, a=REF.a_round_trip, tcoup=REF.t_coupler, L=L_RT, lam0=LAM):
    n = neff0 - (lam - lam0) * (ng - neff0) / lam0            # first-order dispersion, n_g fixed
    phi = 2 * np.pi * n * L / lam
    return np.abs((tcoup - a * np.exp(-1j * phi)) / (1 - tcoup * a * np.exp(-1j * phi))) ** 2

def resonances(lam, T):
    idx = np.where((T[1:-1] < T[:-2]) & (T[1:-1] < T[2:]))[0] + 1
    return lam[idx]

m_ref = REF.neff * L_RT / LAM
m_solver = neff_ref * L_RT / LAM
combs = {}
for lab, ne, ng in (("textbook n_eff 2.5, n_g 4.2", REF.neff, REF.ng), (f"solver n_eff {neff_ref:.3f}, n_g 4.2", neff_ref, REF.ng),
                    (f"solver n_eff {neff_ref:.3f}, solver n_g {ng_solver:.3f}", neff_ref, ng_solver)):
    T = allpass(lam_axis, ne, ng)
    r = resonances(lam_axis, T)
    combs[lab] = (T, r, np.diff(r).mean() * 1e3)
    say(f"  {lab}: resonances at {np.round(r*1e3,2)} nm, FSR {combs[lab][2]:.2f} nm")
say(f"  mode order m = n_eff L/lambda: textbook {m_ref:.1f}, solver {m_solver:.1f} (nearest integer resonances differ; FSR does not)")

fig, axes = plt.subplots(len(combs), 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
for ax, (lab, (T, r, fsr)), c in zip(axes, combs.items(), SERIES):
    ax.plot(lam_axis * 1e3, T, color=c, lw=1.2)
    for rr in r:
        ax.axvline(rr * 1e3, color=c, lw=0.5, ls=":")
    ax.axvline(LAM * 1e3, color=PALETTE["ink2"], lw=1, ls="--")
    ax.set_ylabel("through T"); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{lab}: FSR = {fsr:.2f} nm, resonances at " + ", ".join(f"{x*1e3:.2f}" for x in r) + " nm", fontsize=9.5)
axes[-1].set_xlabel("wavelength (nm)")
fig.suptitle("All-pass ring (a = t = 0.945, L = 39.6 um): n_eff sets WHERE the comb sits, n_g sets its SPACING (dashed = 1310 nm laser)")
fig.savefig(OUT / "08_neff_vs_ng_comb.png")
plt.close(fig)

# ================================================================== G. video: mode vs width
say("\n--- G. video of the TE0 mode morphing with width")
t = time.time()
HOLD = 14
nfr = len(frames) * HOLD
fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
e2max = max(f[4].max() for f in frames)
exmax = max(np.abs(f[3]).max() for f in frames)
iml = axl.imshow(frames[0][4].T / e2max, origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1, aspect="equal")
imr = axr.imshow(frames[0][3].T / exmax, origin="lower", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
rects = [Rectangle((-frames[0][0] / 2, -H / 2), frames[0][0], H, fill=False, ec=PALETTE["ink"], lw=1, ls="--") for _ in range(2)]
axl.add_patch(rects[0]); axr.add_patch(rects[1])
for ax, ttl in ((axl, "|E|^2 (normalised to the 700 nm peak)"), (axr, "E_x (signed)")):
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.45, 0.45); ax.grid(False); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)"); ax.set_title(ttl, fontsize=10)
ttl = fig.suptitle("")

def draw(k):
    w, ne, gP, ex, e2 = frames[min(k // HOLD, len(frames) - 1)]
    iml.set_data(e2.T / e2max); imr.set_data(ex.T / exmax)
    for r in rects:
        r.set_bounds(-w / 2, -H / 2, w, H)
    ttl.set_text(f"TE0 mode (femwell) vs strip width: w = {w*1e3:.0f} nm, n_eff = {ne:.3f}, Gamma_P = {gP:.2f}  (1310 nm, 220 nm thick)")
    return iml, imr, ttl

anim = FuncAnimation(fig, draw, frames=nfr, interval=1000 / 30, blit=False)
anim.save(OUT / "08_mode_vs_width.mp4", writer=FFMpegWriter(fps=30, bitrate=2400))
plt.close(fig)
# contact sheet
picks = [0, 4, 8, 10, 15, 20]
fig, axes = plt.subplots(2, len(picks), figsize=(3.0 * len(picks), 5.2), constrained_layout=True)
for j, i in enumerate(picks):
    w, ne, gP, ex, e2 = frames[i]
    axes[0, j].imshow(e2.T / e2max, origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axes[1, j].imshow(ex.T / exmax, origin="lower", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for ax in axes[:, j]:
        ax.add_patch(Rectangle((-w / 2, -H / 2), w, H, fill=False, ec=PALETTE["ink"], lw=0.8, ls="--"))
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.45, 0.45); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    axes[0, j].set_title(f"w = {w*1e3:.0f} nm\nn_eff = {ne:.3f}, Gamma_P = {gP:.2f}", fontsize=9)
axes[0, 0].set_ylabel("|E|^2"); axes[1, 0].set_ylabel("E_x")
fig.suptitle("Frames of 08_mode_vs_width.mp4: a narrow strip pushes the mode out into the cladding (low n_eff, low Gamma), a wide one swallows it")
fig.savefig(OUT / "08_mode_vs_width_frames.png")
plt.close(fig)
say(f"(G took {time.time()-t:.1f} s)")

# ================================================================== H. results, tools
runtime = time.time() - T0
say(f"\nTotal runtime {runtime:.0f} s")

results = {
    "inputs": dict(lambda_nm=REF.lambda_nm, width_nm=W * 1e3, height_nm=H * 1e3, n_si=NSI, n_sio2=NSIO2,
                   round_trip_um=L_RT, tidy3d_grid_um=TD_DL, tidy3d_plane_um=list(TD_BOX),
                   femwell_core_mesh_um=FEM_RES_REF, femwell_triangles=fm_ref.nelements),
    "reference_modes": {s: [dict(label=m.label, n_eff=m.n_eff, te_fraction=m.te_fraction, gamma_P=m.gamma_P,
                                 gamma_E=m.gamma_E, S_si=m.S_core, S_sio2=m.S_clad, n_g_wg=m.n_g_wg)
                            for m in ms] for s, ms in (("tidy3d", td_ref), ("femwell", fw_ref))},
    "te0": dict(n_eff_tidy3d=te0_td.n_eff, n_eff_femwell=te0_fw.n_eff, n_eff_mean=neff_ref,
                solver_spread_pct=pct(te0_td.n_eff, te0_fw.n_eff), textbook_neff=REF.neff,
                textbook_vs_solver_pct=pct(REF.neff, neff_ref), slab_neff_exp07=2.98795,
                lambda_g_nm=LAM / neff_ref * 1e3, lambda_g_textbook_nm=LAM / REF.neff * 1e3,
                te_fraction_tidy3d=te0_td.te_fraction, te_fraction_femwell=te0_fw.te_fraction,
                gamma_P_tidy3d=te0_td.gamma_P, gamma_P_femwell=te0_fw.gamma_P, gamma_P_ref=REF.confinement,
                gamma_P_vs_ref_pct=pct(0.5 * (te0_td.gamma_P + te0_fw.gamma_P), REF.confinement),
                gamma_E_tidy3d=te0_td.gamma_E, gamma_E_femwell=te0_fw.gamma_E,
                S_si_tidy3d=te0_td.S_core, S_si_femwell=te0_fw.S_core, S_sio2_femwell=te0_fw.S_clad,
                decay_length_1_over_gamma_nm=gamma_ref * 1e3, tail_fit_nm_tidy3d=tail_td * 1e3, tail_fit_nm_femwell=tail_fw * 1e3,
                sidewall_Ex_jump_10nm_tidy3d=jump_td, sidewall_Ex_jump_10nm_femwell=jump_fw,
                sidewall_Ex_jump_at_wall_femwell=jump_wall, eps_ratio=(NSI / NSIO2) ** 2,
                sidewall_jump_vs_eps_ratio_pct=pct(jump_wall, (NSI / NSIO2) ** 2),
                ez_over_ex_peak_tidy3d=ez_over_ex_td, ez_over_ex_peak_femwell=ez_over_ex_fw,
                mode_order_m_solver=m_solver, mode_order_m_textbook=m_ref),
    "group_index": dict(n_g_wg_tidy3d=ng_wg_td, n_g_wg_femwell=ng_wg_fw, n_g_total_tidy3d=ng_tot_td, n_g_total_femwell=ng_tot_fw,
                        n_g_total_predicted_tidy3d=ng_pred_td, n_g_total_predicted_femwell=ng_pred_fw,
                        n_g_total_mean=ng_solver, n_g_ref=REF.ng, n_g_vs_ref_pct=pct(ng_solver, REF.ng),
                        material_term_femwell=mat_term_fw, waveguide_term_femwell=ng_wg_fw - te0_fw.n_eff,
                        n_g_si_bulk=ng_si_bulk, n_g_sio2_bulk=ng_sio2_bulk,
                        fsr_nm_solver=fsr_nm_solver, fsr_thz_solver=fsr_thz_solver, fsr_nm_ref=REF.fsr_nm, fsr_thz_ref=REF.fsr_thz,
                        fsr_vs_ref_pct=pct(fsr_nm_solver, REF.fsr_nm)),
    "width_sweep": {k: v.tolist() for k, v in sw.items()} | dict(
        te1_cutoff_width_nm_tidy3d=cut_td * 1e3, te1_cutoff_width_nm_femwell=cut_fw * 1e3,
        neff_2p5_crossing_width_nm=neff_2p5_w, ng_4p2_crossing_width_nm=ng_4p2_w,
        dneff_dw_per_nm_tidy3d=dneff_dw_td, dneff_dw_per_nm_femwell=dneff_dw_fw,
        dlambda_r_per_nm_width_pm=dlam_dw_fw * 1e3, shift_for_10nm_width_error_nm=dlam_dw_fw * 10,
        shift_for_10nm_width_error_in_FSR=dlam_dw_fw * 10 / fsr_nm_solver),
    "wavelength_sweep": {k: v.tolist() for k, v in wl.items()} | dict(channel_wavelengths_nm=(chan_lam * 1e3).tolist(),
                                                                       channel_spacing_nm=abs(chan_lam[1] - chan_lam[0]) * 1e3),
    "thermo_optic": dict(dT_K=DT, dn_si=dnsi, dn_sio2=dnsio2, per_solver=res_to,
                         femwell_linearity=dict(dT=dTs.tolist(), n_eff=neff_T.tolist(), slope_per_K=lin[0]),
                         dneff_dT_over_dnsi_dT=lin[0] / REF.dn_si_dT,
                         dlambda_dT_pm_per_K=dict(ref=REF.dlambda_dT_pm_per_K, textbook_gamma_weighted=textbook_est,
                                                  perturbation_formula_ng_ref=pert_est,
                                                  tidy3d_solver_ng=shift["tidy3d"]["ng_solver"], femwell_solver_ng=shift["femwell"]["ng_solver"],
                                                  tidy3d_ng_ref=shift["tidy3d"]["ng_ref"], femwell_ng_ref=shift["femwell"]["ng_ref"],
                                                  solver_mean=dldT, solver_mean_vs_ref_pct=pct(dldT, REF.dlambda_dT_pm_per_K),
                                                  textbook_vs_ref_pct=pct(textbook_est, REF.dlambda_dT_pm_per_K)),
                         one_K_pm=dldT, one_K_fraction_of_fwhm=one_K_frac, tenth_K_pm=dldT / 10,
                         delta_opt_in_K=REF.delta_opt_pm / dldT, ambient_span_K=amb, ambient_drift_nm=amb * dldT * 1e-3,
                         ambient_drift_in_FSR=amb * dldT * 1e-3 / fsr_nm_solver,
                         K_per_channel_spacing=abs(chan_lam[1] - chan_lam[0]) * 1e6 / dldT,
                         K_per_FSR=fsr_nm_solver * 1e3 / dldT,
                         thermal_expansion=dict(alpha_L_per_K=ALPHA_L_SI, neff_alpha_L_per_K=exp_term_index,
                                                fraction_of_dneff_dT=exp_frac, term_pm_per_K=exp_term_pm,
                                                dlambda_dT_total_pm_per_K=dldT_total,
                                                total_vs_ref_pct=pct(dldT_total, REF.dlambda_dT_pm_per_K),
                                                cross_section_widening_per_K=geom_term_index,
                                                cross_section_fraction_of_dneff_dT=geom_term_index / dneff_dT_mean)),
    "ring_comb": {lab: dict(resonances_nm=(r * 1e3).tolist(), fsr_nm=f) for lab, (T, r, f) in combs.items()},
    "runtime_s": runtime,
}
(OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))

def ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "?"

ffv = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0].split(" ")[2]
tools = [
    {"tool": "Tidy3D local mode solver (tidy3d.plugins.mode.ModeSolver)", "version": ver("tidy3d"),
     "what_it_is": "Tidy3D is Flexcompute's FDTD package; its ModeSolver plugin is a finite-difference frequency-domain (FDFD) eigenmode solver on a Yee grid that runs locally without an account. It is normally used to define mode sources/monitors for FDTD runs and to get n_eff, n_g and mode profiles of waveguide cross-sections.",
     "used_for": f"Full-vector modes of the 500x220 nm SOI strip (uniform {TD_DL*1e3:.0f} nm Yee grid on a {TD_BOX[0]}x{TD_BOX[1]} um plane, sized so the core edges fall mid-cell because the local solver has no subpixel averaging): n_eff, group index by its built-in frequency step, TE fraction, Poynting/energy confinement from its field arrays, the width and wavelength sweeps and the thermo-optic perturbation (re-solve with n_Si + 1.86e-4 K^-1 x 10 K).",
     "result": f"TE0 n_eff = {te0_td.n_eff:.4f} (femwell {te0_fw.n_eff:.4f}, spread {pct(te0_td.n_eff, te0_fw.n_eff):+.2f} %; textbook 2.5 is {pct(REF.neff, neff_ref):+.1f} % off), TE = {te0_td.te_fraction:.3f}, Gamma_P = {te0_td.gamma_P:.3f} (REF 0.85), n_g,wg = {ng_wg_td:.3f}, n_g total = {ng_tot_td:.3f} (REF 4.2, {pct(ng_tot_td, REF.ng):+.1f} %), dn_eff/dT = {res_to['tidy3d']['dneff_dT']:.3e}/K -> {shift['tidy3d']['ng_solver']:.1f} pm/K (REF 50, {pct(shift['tidy3d']['ng_solver'], 50):+.0f} %).",
     "how_to_observe": "cd experiments/08_soi_strip_mode_solvers && ../../.venv/bin/python run.py ; look at out/08_mode_fields.png (top row), out/08_width_sweep.png, out/08_wavelength_sweep.png, out/08_thermo_optic.png; numbers in out/results.json. Change TD_DL (grid) or TD_BOX in run.py, or the width/height/indices passed to solve_tidy3d in solvers.py, and re-run."},
    {"tool": "femwell (FEM Maxwell mode solver on scikit-fem + gmsh)", "version": ver("femwell") + f" (scikit-fem {ver('skfem')}, gmsh {ver('gmsh')}, shapely {ver('shapely')}, meshio {ver('meshio')})",
     "what_it_is": "femwell is an open-source finite-element photonics package built on scikit-fem: it meshes 2-D cross-sections with gmsh (triangles that conform to the material boundaries) and solves the vector Maxwell eigenproblem with Nedelec (transverse) and Lagrange (longitudinal) elements. Used for waveguide modes, coupling coefficients, thermal and electro-optic overlaps.",
     "used_for": f"The same strip on a boundary-conforming mesh ({fm_ref.nelements} triangles, {FEM_RES_REF*1e3:.0f} nm elements in the core, 2nd-order elements): n_eff, TE fraction, exact Poynting power fraction in Si (calculate_power on the core elements), the perturbation sensitivity dn_eff/dn_Si (its calculate_confinement_factor), group index by finite difference in wavelength, the width sweep (also feeds the video), the 0-115 K linearity check and the thermo-optic re-solve.",
     "result": f"TE0 n_eff = {te0_fw.n_eff:.4f} (converged to 1e-5 between 20 nm and 5 nm meshes), TE = {te0_fw.te_fraction:.3f}, Gamma_P = {te0_fw.gamma_P:.3f} (REF 0.85, {pct(te0_fw.gamma_P, 0.85):+.1f} %), S_Si = dn_eff/dn_Si = {te0_fw.S_core:.3f} (= direct re-solve {res_to['femwell']['S_si_direct']:.3f}), n_g,wg = {ng_wg_fw:.3f}, n_g total = {ng_tot_fw:.3f} (REF 4.2, {pct(ng_tot_fw, REF.ng):+.1f} %), dlambda_r/dT = {shift['femwell']['ng_solver']:.1f} pm/K (REF 50, {pct(shift['femwell']['ng_solver'], 50):+.0f} %), TE1 cutoff width ~{cut_fw*1e3:.0f} nm.",
     "how_to_observe": "Same command; out/08_mode_fields.png (bottom row), out/08_mode_gallery.png, out/08_profile_cuts.png, out/08_mode_vs_width.mp4 and its frames PNG. Change FEM_RES_REF/FEM_RES_SWEEP or the box in femwell_mesh (solvers.py) and re-run; the mesh is rebuilt by gmsh each time."},
    {"tool": "numpy / scipy / matplotlib", "version": f"numpy {ver('numpy')}, scipy {ver('scipy')}, matplotlib {ver('matplotlib')}",
     "what_it_is": "The scientific Python stack: arrays and linear algebra (numpy, also the ARPACK eigensolver femwell calls through scipy), plotting and FuncAnimation video (matplotlib).",
     "used_for": "Sellmeier evaluation of n_Si(lambda), n_SiO2(lambda) and their bulk group indices (materials.py); finite-difference group indices; tail-decay fit; the all-pass ring comb T(lambda) of notes 28 for the n_eff-versus-n_g panel; every figure and the mp4.",
     "result": f"bulk n_g,Si = {ng_si_bulk:.3f} at 1310 nm (adds {mat_term_fw:+.3f} to the waveguide n_g); ring comb FSR = {combs[list(combs)[0]][2]:.2f} nm with n_eff = 2.5 and {combs[list(combs)[1]][2]:.2f} nm with n_eff = {neff_ref:.3f}, both at n_g = 4.2; {combs[list(combs)[2]][2]:.2f} nm at the solver n_g {ng_solver:.3f} (FSR is n_g's job, n_eff only slides the comb).",
     "how_to_observe": "out/08_neff_vs_ng_comb.png, out/08_profile_cuts.png; python materials.py prints the bulk indices."},
    {"tool": "ffmpeg", "version": ffv,
     "what_it_is": "Command-line video encoder/decoder; matplotlib's FFMpegWriter pipes frames to it.",
     "used_for": "Encoding out/08_mode_vs_width.mp4 (H.264, 30 fps, ~10 s).",
     "result": f"{nfr} frames from {len(frames)} femwell solutions between 300 and 700 nm width.",
     "how_to_observe": "open out/08_mode_vs_width.mp4; stills in out/08_mode_vs_width_frames.png; change HOLD or the widths array in run.py."},
]
(OUT / "tools.json").write_text(json.dumps(tools, indent=1))
(OUT / "results.txt").write_text("\n".join(LOG) + "\n")
say("wrote out/results.json, out/tools.json, out/results.txt")
