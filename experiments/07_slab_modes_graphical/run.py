"""07_slab_modes_graphical: TE modes of the symmetric slab, solved graphically and checked with Meep.

Run with the Meep interpreter from this directory:
    cd experiments/07_slab_modes_graphical && ../../.meep/bin/python run.py

Everything analytic lives in slab_te.py (numpy/scipy only, shared with explore.ipynb).
Notes sections implemented: 14-19 (eigenproblem, even/odd equations), 21 (two-plane-wave
decomposition), 22 (n_eff, lambda_g, n_g), 23 (the 220 nm slab).
"""
import sys, pathlib, json, time, subprocess, shutil
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import REF, use_style, SERIES, PALETTE
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyArrowPatch
import numpy as np
import meep as mp

import slab_te as st

T_START = time.time()
mp.verbosity(0)

# ----------------------------------------------------------------------------- reference case
LAM = REF.lambda_nm * 1e-3          # um
N1, N2 = REF.n_si, REF.n_sio2
THICK = REF.wg_height_um            # 2d = 0.22 um
D = THICK / 2
K0 = st.k0_per_um(LAM)
RESULTS = {"inputs": {"lambda_nm": REF.lambda_nm, "n1_si": N1, "n2_sio2": N2, "thickness_nm": THICK * 1e3}}

modes = st.solve_te_modes(D, LAM, N1, N2)
V = st.V_number(D, LAM, N1, N2)
print(f"Slab 2d = {THICK*1e3:.0f} nm, lambda = {LAM*1e3:.0f} nm, n1 = {N1}, n2 = {N2}: V = {V:.4f} -> {len(modes)} guided TE modes")
RESULTS["V"] = V
RESULTS["n_guided_te_modes"] = len(modes)
RESULTS["modes_analytic"] = []
for md in modes:
    print(f"  TE{md.m} ({md.parity}): hd = {md.hd:.4f}, gd = {md.gd:.4f}, beta = {md.beta_per_um:.4f} rad/um, "
          f"n_eff = {md.neff:.5f}, 1/gamma = {md.decay_len_um*1e3:.1f} nm, lambda_g = {md.lambda_g_um*1e3:.1f} nm, "
          f"theta = {md.theta_deg:.2f} deg, Gamma = {md.confinement:.3f}")
    RESULTS["modes_analytic"].append(md.as_dict())

te0 = modes[0]
RESULTS["te0"] = {
    "neff": te0.neff, "beta_rad_per_um": te0.beta_per_um, "gamma_per_um": te0.gamma_per_um,
    "decay_length_nm": te0.decay_len_um * 1e3, "lambda_g_nm": te0.lambda_g_um * 1e3,
    "theta_incidence_deg": te0.theta_deg, "confinement": te0.confinement,
    "cosine_cycles_in_core": 2 * te0.hd / (2 * np.pi),
    "F_at_boundary_over_F_center": float(np.cos(te0.hd)),
}
RESULTS["critical_angle_deg"] = float(np.degrees(np.arcsin(N2 / N1)))
RESULTS["notes_expectation"] = {
    "neff_notes_sec22": 2.97, "decay_length_nm_notes_sec23": 80.0, "lambda_g_nm_notes_sec22": 441.0,
    "neff_ref_strip_textbook": REF.neff, "ng_ref_strip_textbook": REF.ng,
}
RESULTS["te0"]["neff_vs_notes_sec22_pct"] = 100 * (te0.neff / 2.97 - 1)
RESULTS["te0"]["decay_length_vs_notes_sec23_pct"] = 100 * (te0.decay_len_um * 1e3 / 80.0 - 1)
RESULTS["te0"]["lambda_g_vs_notes_sec22_pct"] = 100 * (te0.lambda_g_um * 1e3 / 441.0 - 1)

# cutoffs
RESULTS["cutoff"] = {
    "te1_thickness_nm_at_1310": st.cutoff_thickness_um(1, LAM, N1, N2) * 1e3,
    "te2_thickness_nm_at_1310": st.cutoff_thickness_um(2, LAM, N1, N2) * 1e3,
    "te1_wavelength_nm_at_220nm": st.cutoff_wavelength_um(1, D, N1, N2) * 1e3,
    "formula": "TE_m guided when V = k0 d sqrt(n1^2-n2^2) > m pi/2, i.e. 2d > m lambda / (2 sqrt(n1^2-n2^2))",
}
print("cutoffs:", RESULTS["cutoff"])

# group index (waveguide dispersion only, materials fixed)
ng_te0 = st.group_index(D, LAM, N1, N2, 0)
ng_te1 = st.group_index(D, LAM, N1, N2, 1)
RESULTS["te0"]["ng_waveguide_dispersion_only"] = ng_te0
RESULTS["te1_ng_waveguide_dispersion_only"] = ng_te1
print(f"n_g (fixed materials): TE0 {ng_te0:.4f}, TE1 {ng_te1:.4f}; REF strip n_g = {REF.ng}")

# ----------------------------------------------------------------------------- "what if" cases quoted in README's Experiments to try
def what_if(thick_um=THICK, lam_um=LAM, n1=N1, n2=N2):
    ms = st.solve_te_modes(thick_um / 2, lam_um, n1, n2)
    return {"thickness_nm": thick_um * 1e3, "lambda_nm": lam_um * 1e3, "n1": n1, "n2": n2,
            "V": st.V_number(thick_um / 2, lam_um, n1, n2), "n_guided_te_modes": len(ms),
            "critical_angle_deg": float(np.degrees(np.arcsin(n2 / n1))),
            "modes": [{"m": m.m, "neff": m.neff, "decay_length_nm": m.decay_len_um * 1e3,
                       "confinement": m.confinement, "theta_deg": m.theta_deg} for m in ms]}
RESULTS["what_if"] = {
    "thickness_200nm": what_if(thick_um=0.20),
    "thickness_450nm": what_if(thick_um=0.45),
    "lambda_1450nm": what_if(lam_um=1.45),
    "lambda_1000nm": what_if(lam_um=1.00),
    "n1_2p0_nitride_like": what_if(n1=2.0),
    "n2_1p0_air_cladding": what_if(n2=1.0),
}
for key, w in RESULTS["what_if"].items():
    m0 = w["modes"][0]
    print(f"  what-if {key:<22}: V = {w['V']:.3f}, {w['n_guided_te_modes']} TE mode(s); TE0 n_eff {m0['neff']:.3f}, "
          f"1/γ {m0['decay_length_nm']:.0f} nm, Γ {m0['confinement']:.2f}, θ {m0['theta_deg']:.1f}° (θ_c {w['critical_angle_deg']:.1f}°)"
          + (f"; TE{w['modes'][-1]['m']} 1/γ {w['modes'][-1]['decay_length_nm']:.0f} nm" if len(w["modes"]) > 1 else ""))

# ----------------------------------------------------------------------------- thermo-optic sensitivity (capstone bridge)
# dn_eff/dT = (dn_eff/dn1) dn1/dT + (dn_eff/dn2) dn2/dT, each partial from a centred finite difference of the exact root.
# First-order perturbation theory for the slab gives the exact identities
#     dn_eff/dn1 = (n_g/n1) Gamma_E = (n1/n_eff) Gamma_F,     dn_eff/dn2 = (n_g/n2) (1 - Gamma_E),
# with Gamma_F = int_core F^2 / int F^2 (the |F|^2, i.e. power, fraction) and Gamma_E = int_core n^2 F^2 / int n^2 F^2
# (the electric-energy fraction).  Both forms agree because n_eff n_g = <n^2>_F = n1^2 Gamma_F + n2^2 (1 - Gamma_F).
# Putting the n_g form into dlambda_r/dT = (lambda/n_g) dn_eff/dT cancels n_g:
#     dlambda_r/dT = lambda [Gamma_E (dn1/dT)/n1 + (1 - Gamma_E) (dn2/dT)/n2]      (n_g-free)
# so the only mode quantity in the thermal drift is the silicon energy fraction Gamma_E.
# Textbook shortcut (REF): dn_eff/dT ~ Gamma dn_Si/dT with Gamma = 0.85 and dlambda/dT = (lambda/n_g) dn_eff/dT, n_g = 4.2 -> 49 pm/K.
def neff_te0(n1, n2, lam=LAM, d=D):
    return st.solve_te_modes(d, lam, n1, n2)[0].neff
dn = 1e-4
dneff_dn1 = (neff_te0(N1 + dn, N2) - neff_te0(N1 - dn, N2)) / (2 * dn)
dneff_dn2 = (neff_te0(N1, N2 + dn) - neff_te0(N1, N2 - dn)) / (2 * dn)
gamma_F = te0.confinement
gamma_E = te0.energy_fraction
# numerical cross-check of both fractions by trapezoid integration of the analytic profile
x_int = np.linspace(-3.0, 3.0, 600_001)
F_int, _ = st.profile(te0, x_int)
n_int = np.where(np.abs(x_int) <= D, N1, N2)
core_mask = np.abs(x_int) <= D
gamma_F_num = np.trapezoid(F_int[core_mask]**2, x_int[core_mask]) / np.trapezoid(F_int**2, x_int)
gamma_E_num = np.trapezoid((n_int * F_int)[core_mask]**2, x_int[core_mask]) / np.trapezoid((n_int * F_int)**2, x_int)
ident_ng = ng_te0 / N1 * gamma_E                    # (n_g/n1) Gamma_E
ident_neff = N1 / te0.neff * gamma_F                # (n1/n_eff) Gamma_F
ident_clad = ng_te0 / N2 * (1 - gamma_E)            # (n_g/n2) (1 - Gamma_E)
mean_n2_F = N1**2 * gamma_F + N2**2 * (1 - gamma_F)   # <n^2>_F, must equal n_eff n_g
dneff_dT = dneff_dn1 * REF.dn_si_dT + dneff_dn2 * REF.dn_sio2_dT
dneff_dT_gamma_rule = gamma_F * REF.dn_si_dT + (1 - gamma_F) * REF.dn_sio2_dT
dneff_dT_textbook = REF.confinement * REF.dn_si_dT
LAM_PM = LAM * 1e6                                   # 1310 nm in pm
dlam_dT_slab_ng = LAM_PM / ng_te0 * dneff_dT         # pm/K, via (lambda/n_g) dn_eff/dT with the FD partials
dlam_dT_si_part = LAM_PM * gamma_E * REF.dn_si_dT / N1             # closed form, silicon term (n_g-free)
dlam_dT_clad_part = LAM_PM * (1 - gamma_E) * REF.dn_sio2_dT / N2   # closed form, silica term
dlam_dT_closed = dlam_dT_si_part + dlam_dT_clad_part
dlam_dT_textbook = LAM_PM / REF.ng * dneff_dT_textbook
# what silicon energy fraction would a waveguide need to sit at the REF 50 pm/K?
a_si = LAM_PM * REF.dn_si_dT / N1; a_clad = LAM_PM * REF.dn_sio2_dT / N2
gamma_E_for_ref_si_only = REF.dlambda_dT_pm_per_K / a_si
gamma_E_for_ref_with_clad = (REF.dlambda_dT_pm_per_K - a_clad) / (a_si - a_clad)
RESULTS["thermo_optic"] = {
    "dneff_dn1_finite_difference": dneff_dn1, "dneff_dn2_finite_difference": dneff_dn2,
    "Gamma_F_power_fraction_closed_form": gamma_F, "Gamma_F_power_fraction_numeric": gamma_F_num,
    "Gamma_E_energy_fraction_closed_form": gamma_E, "Gamma_E_energy_fraction_numeric": gamma_E_num,
    "identity_ng_over_n1_times_Gamma_E": ident_ng,
    "identity_n1_over_neff_times_Gamma_F": ident_neff,
    "identity_ng_over_n2_times_1_minus_Gamma_E": ident_clad,
    "identity_neff_times_ng": te0.neff * ng_te0, "identity_mean_n2_weighted_by_F2": mean_n2_F,
    "n1_over_neff": N1 / te0.neff, "ng_over_n1": ng_te0 / N1,
    "dneff_dT_per_K_exact_slab": dneff_dT,
    "dneff_dT_per_K_Gamma_F_rule_slab": dneff_dT_gamma_rule,
    "dneff_dT_per_K_textbook_0p85": dneff_dT_textbook,
    "dlambda_dT_pm_per_K_slab_via_lambda_over_ng": dlam_dT_slab_ng,
    "dlambda_dT_pm_per_K_slab_closed_form_ng_free": dlam_dT_closed,
    "dlambda_dT_pm_per_K_slab_silicon_term": dlam_dT_si_part,
    "dlambda_dT_pm_per_K_slab_silica_term": dlam_dT_clad_part,
    "dlambda_dT_pm_per_K_textbook": dlam_dT_textbook,
    "dlambda_dT_pm_per_K_REF": REF.dlambda_dT_pm_per_K,
    "Gamma_E_needed_for_REF_50_pm_per_K_silicon_term_only": gamma_E_for_ref_si_only,
    "Gamma_E_needed_for_REF_50_pm_per_K_with_silica_term": gamma_E_for_ref_with_clad,
    "formula": "dlambda_r/dT = (lambda/n_g) dn_eff/dT with dn_eff/dn1 = (n_g/n1) Gamma_E and dn_eff/dn2 = (n_g/n2)(1 - Gamma_E) "
               "=> dlambda_r/dT = lambda [Gamma_E dn1/dT / n1 + (1 - Gamma_E) dn2/dT / n2]; n_g cancels, only Gamma_E matters",
}
assert abs(ident_ng / dneff_dn1 - 1) < 1e-4 and abs(ident_neff / dneff_dn1 - 1) < 1e-4, "perturbation identity broken"
assert abs(ident_clad / dneff_dn2 - 1) < 1e-3, "cladding perturbation identity broken"
assert abs(dlam_dT_closed / dlam_dT_slab_ng - 1) < 1e-4, "n_g did not cancel"
print(f"thermo-optic: dn_eff/dn1 = {dneff_dn1:.5f} (FD) = (n_g/n1) Gamma_E = {ident_ng:.5f} = (n1/n_eff) Gamma_F = {ident_neff:.5f}; "
      f"Gamma_F = {gamma_F:.4f} (numeric {gamma_F_num:.4f}), Gamma_E = {gamma_E:.4f} (numeric {gamma_E_num:.4f}); "
      f"dn_eff/dn2 = {dneff_dn2:.5f} (FD) = (n_g/n2)(1-Gamma_E) = {ident_clad:.5f}; n_eff n_g = {te0.neff*ng_te0:.4f} = <n^2>_F = {mean_n2_F:.4f}")
print(f"thermo-optic: dn_eff/dT = {dneff_dT:.3e}/K (Gamma_F rule {dneff_dT_gamma_rule:.3e}, textbook {dneff_dT_textbook:.3e}); "
      f"dlambda/dT = {dlam_dT_slab_ng:.2f} pm/K via lambda/n_g = {dlam_dT_closed:.2f} pm/K closed form "
      f"(Si {dlam_dT_si_part:.2f} + SiO2 {dlam_dT_clad_part:.2f}, n_g-free); textbook {dlam_dT_textbook:.1f}, REF {REF.dlambda_dT_pm_per_K}; "
      f"a strip at 50 pm/K needs Gamma_E = {gamma_E_for_ref_with_clad:.3f} ({gamma_E_for_ref_si_only:.3f} ignoring the silica term)")


# ----------------------------------------------------------------------------- figure 1: graphical construction
def draw_construction(ax, d_um, lam_um, n1, n2, title):
    Vn = st.V_number(d_um, lam_um, n1, n2)
    ms = st.solve_te_modes(d_um, lam_um, n1, n2)
    umax = Vn + 0.3
    ymax = Vn + 0.6
    u = np.linspace(1e-3, umax, 4000)
    even = u * np.tan(u)
    odd = -u / np.tan(u)
    even = np.where((even >= 0) & (even < ymax * 1.05), even, np.nan)
    odd = np.where((odd >= 0) & (odd < ymax * 1.05), odd, np.nan)
    # break the curves at the poles so they do not connect across branches
    for arr in (even, odd):
        pole = np.abs(np.diff(arr)) > 0.5
        arr[1:][pole] = np.nan
    ax.plot(u, even, color=SERIES[0], label="even: hd·tan(hd)")
    ax.plot(u, odd, color=SERIES[1], label="odd: −hd·cot(hd)")
    th = np.linspace(0, np.pi / 2, 300)
    ax.plot(Vn * np.cos(th), Vn * np.sin(th), color=SERIES[2], label=f"circle: (hd)²+(γd)² = V², V = {Vn:.3f}")
    for md in ms:
        ax.plot(md.hd, md.gd, "o", color=SERIES[3], ms=8, zorder=5)
        ax.annotate(f"TE{md.m}\nn_eff={md.neff:.3f}", (md.hd, md.gd), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=PALETTE["ink"])
    for k in range(1, int(umax / (np.pi / 2)) + 1):
        ax.axvline(k * np.pi / 2, color=PALETTE["line"], lw=0.8, ls="--")
    ax.set_xlim(0, umax); ax.set_ylim(0, ymax)
    ax.set_xlabel("hd  (transverse phase across half the core, rad)")
    ax.set_ylabel("γd  (cladding decay rate × half thickness)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")


fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6))
draw_construction(axs[0], D, LAM, N1, N2,
                  f"220 nm slab at 1310 nm: the circle (V = {V:.2f}) meets one even\nand one odd branch, so exactly 2 TE modes are guided")
draw_construction(axs[1], 0.25, LAM, N1, N2,
                  "500 nm slab at 1310 nm: a larger V circle cuts more branches\n(3 TE modes); TE3 would need V > 3π/2")
fig.suptitle("Graphical solution of h·tan(hd) = γ (even) and −h·cot(hd) = γ (odd) with (hd)² + (γd)² = V²  [notes 19]", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "07_graphical_construction.png"); plt.close(fig)

# ----------------------------------------------------------------------------- figure 2: profiles F, Hx, Hz
x = np.linspace(-0.9, 0.9, 3601)
fig, axs = plt.subplots(3, 2, figsize=(12, 9.5), sharex="col")
for col, md in enumerate(modes[:2]):
    Ey, Hx, Hz = st.fields(md, x, E0=1.0)      # E0 = 1 V/m -> H in A/m
    F, dF = st.profile(md, x)
    span = 0.45 if md.m == 0 else 0.9          # TE0's 80 nm tail needs a tighter axis to be readable
    for row, (y, lab, ttl) in enumerate([
        (Ey, "E_y = F(x)  (V/m per V/m)", "E_y = F(x): cosine (even) or sine (odd) in the core, exponential tail outside"),
        (Hx * 1e3, "H_x = −(β/ωμ)·F  (mA/m)", "H_x follows F(x) exactly (same shape, opposite sign)"),
        (Hz.imag * 1e3, "Im H_z = F′/(ωμ)  (mA/m)", "H_z ∝ F′(x): continuous at x = ±d because tangential H is continuous [notes 18]"),
    ]):
        ax = axs[row, col]
        ax.axvspan(-md.d_um, md.d_um, color=SERIES[0], alpha=0.10, lw=0)
        ax.plot(x, y, color=SERIES[col])
        ax.axvline(-md.d_um, color=PALETTE["muted"], lw=0.8, ls="--"); ax.axvline(md.d_um, color=PALETTE["muted"], lw=0.8, ls="--")
        ax.axhline(0, color=PALETTE["line"], lw=0.8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_xlim(-span, span)
        if row == 0:
            ax.set_title(f"TE{md.m} ({md.parity}): n_eff = {md.neff:.4f}, 1/γ = {md.decay_len_um*1e3:.0f} nm, Γ = {md.confinement:.2f}\n{ttl}", fontsize=9)
            # decay-length marker on the right tail: double arrow from x = d to x = d + 1/γ at the height F(d)/e,
            # where the curve reaches at the arrow's far end; the label sits in empty space (above-right for the
            # short TE0 tail, below-left for the long TE1 tail) with a white box so it never overprints the curve.
            xd = md.d_um + md.decay_len_um
            Fd = F[np.argmin(abs(x - md.d_um))]
            ax.annotate("", xy=(xd, Fd / np.e), xytext=(md.d_um, Fd / np.e),
                        arrowprops=dict(arrowstyle="<->", color=PALETTE["ink2"], lw=1.2, shrinkA=0, shrinkB=0))
            ax.axvline(xd, color=PALETTE["ink2"], lw=0.6, ls=":")
            label = f"1/γ = {md.decay_len_um*1e3:.0f} nm: F falls to e⁻¹ of F(d)"
            if xd + 0.35 * span < span:
                ax.text(xd + 0.02, Fd / np.e + 0.05 * abs(y).max(), label, ha="left", va="bottom", fontsize=7.5,
                        color=PALETTE["ink2"], bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
            else:
                ax.text(xd - 0.02, Fd / np.e - 0.05 * abs(y).max(), label, ha="right", va="top", fontsize=7.5,
                        color=PALETTE["ink2"], bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        else:
            ax.set_title(ttl, fontsize=8.5)
        if row == 2:
            ax.set_xlabel("x (µm), core shaded, dashed lines at x = ±d = ±110 nm")
fig.suptitle("Field profiles across the 220 nm slab at 1310 nm: E_y and H_z are both continuous at the boundary, so F and F′ match there", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "07_mode_profiles.png"); plt.close(fig)

# ----------------------------------------------------------------------------- Meep validation (1-D cross-section eigenmode)
def meep_modes(thick_um, lam_um, n1, n2, nmodes_wanted, res=100, cell_y=6.0):
    """Meep 2-D: propagation along Meep x, slab thickness along Meep y, invariant along Meep z.
    Notes' (x, y, z) = Meep's (y, z, x).  Notes' TE (E_y) is Meep's Ez polarisation (ODD_Z)."""
    f = 1 / lam_um
    geom = [mp.Block(size=mp.Vector3(mp.inf, thick_um, mp.inf), material=mp.Medium(index=n1))]
    sim = mp.Simulation(cell_size=mp.Vector3(2, cell_y, 0), geometry=geom, default_material=mp.Medium(index=n2),
                        resolution=res, boundary_layers=[mp.PML(1.0, direction=mp.Y)])
    sim.init_sim()
    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, cell_y, 0))
    out = []
    for m in range(nmodes_wanted):
        parity = (mp.EVEN_Y if m % 2 == 0 else mp.ODD_Y) + mp.ODD_Z
        band = m // 2 + 1        # band index within that parity family
        em = sim.get_eigenmode(f, mp.X, vol, band_num=band, kpoint=mp.Vector3(0.9 * n1 * f, 0, 0),
                               parity=parity, resolution=res, eigensolver_tol=1e-10)
        neff = em.k.x / f
        out.append({"m": m, "neff": float(neff), "beta_rad_per_um": float(2 * np.pi * em.k.x),
                    "ng": float(1 / em.group_velocity) if em.group_velocity else None, "em": em})
    return sim, out


t_meep = time.time()
sim, mm = meep_modes(THICK, LAM, N1, N2, len(modes), res=100)
RESULTS["meep"] = {"resolution_px_per_um": 100, "modes": []}
for md, mr in zip(modes, mm):
    agree = 100 * (mr["neff"] / md.neff - 1)
    ng_agree = 100 * (mr["ng"] / (ng_te0 if md.m == 0 else ng_te1) - 1)
    print(f"  Meep TE{md.m}: n_eff = {mr['neff']:.5f} (analytic {md.neff:.5f}, {agree:+.3f} %), "
          f"beta = {mr['beta_rad_per_um']:.4f} rad/um, n_g = {mr['ng']:.4f} (analytic {ng_te0 if md.m==0 else ng_te1:.4f}, {ng_agree:+.3f} %)")
    RESULTS["meep"]["modes"].append({"m": md.m, "neff_meep": mr["neff"], "neff_analytic": md.neff,
                                     "neff_diff_pct": agree, "beta_meep_rad_per_um": mr["beta_rad_per_um"],
                                     "beta_analytic_rad_per_um": md.beta_per_um,
                                     "ng_meep": mr["ng"], "ng_analytic": ng_te0 if md.m == 0 else ng_te1, "ng_diff_pct": ng_agree})

# resolution convergence for TE0
conv = []
for res in (25, 50, 100, 200):
    _, mm_r = meep_modes(THICK, LAM, N1, N2, 1, res=res)
    conv.append({"resolution": res, "neff": mm_r[0]["neff"], "diff_pct": 100 * (mm_r[0]["neff"] / te0.neff - 1)})
    print(f"  Meep TE0 convergence: res {res:>3} px/um -> n_eff {mm_r[0]['neff']:.5f} ({conv[-1]['diff_pct']:+.3f} %)")
RESULTS["meep"]["te0_resolution_convergence"] = conv

# profile comparison figure
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
for col, (md, mr) in enumerate(zip(modes, mm)):
    em = mr["em"]
    span = 1.0 if md.m == 0 else 2.4
    ys = np.linspace(-span, span, 601)
    Ez = np.array([em.amplitude(mp.Vector3(0, y, 0), mp.Ez) for y in ys])
    Hx_meep = np.array([em.amplitude(mp.Vector3(0, y, 0), mp.Hx) for y in ys])   # = notes' H_z (prop. to F')
    Hy_meep = np.array([em.amplitude(mp.Vector3(0, y, 0), mp.Hy) for y in ys])   # = notes' H_x (prop. to F)
    # normalise Meep's complex amplitudes to a real profile with the same sign convention as F
    ref_idx = np.argmax(np.abs(Ez))
    ph = Ez[ref_idx] / abs(Ez[ref_idx])
    Ez_n = (Ez / ph).real; Hy_n = (Hy_meep / ph).real; Hx_n = (Hx_meep / ph)
    Hx_n = Hx_n.imag if abs(Hx_n.imag).max() > abs(Hx_n.real).max() else Hx_n.real
    F, dF = st.profile(md, ys)
    scale = abs(F).max() / abs(Ez_n).max()
    if np.sign(Ez_n[ref_idx]) != np.sign(F[np.argmin(abs(ys - ys[ref_idx]))]):
        scale = -scale
    ax = axs[0, col]
    ax.axvspan(-D, D, color=SERIES[0], alpha=0.10, lw=0)
    ax.plot(ys, F, color=SERIES[0], label="analytic F(x)")
    ax.plot(ys, Ez_n * scale, "--", color=SERIES[1], label="Meep E (Ez in Meep axes)")
    ax.set_title(f"TE{md.m}: n_eff analytic {md.neff:.4f} vs Meep {mr['neff']:.4f} ({100*(mr['neff']/md.neff-1):+.3f} %)", fontsize=10)
    ax.set_ylabel("E_y (normalised)")
    # short labels keep the box in the empty corner: TE0 upper right (tail < 0.2 beyond 0.3 µm), TE1 upper left (left tail negative)
    ax.legend(fontsize=8, loc="upper right" if md.m == 0 else "upper left")
    ax = axs[1, col]
    ax.axvspan(-D, D, color=SERIES[0], alpha=0.10, lw=0)
    dFn = dF / abs(dF).max()
    hx_s = abs(dFn).max() / abs(Hx_n).max()
    if np.sign(Hx_n[np.argmax(abs(Hx_n))]) != np.sign(dFn[np.argmax(abs(Hx_n))]):
        hx_s = -hx_s
    ax.plot(ys, dFn, color=SERIES[0], label="analytic F′(x) ∝ H_z")
    ax.plot(ys, Hx_n * hx_s, "--", color=SERIES[1], label="Meep H∥z (Hx in Meep axes)")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("F′ (normalised)")
    # TE0: lower-left corner is empty; TE1: the curve is ~0 for x > 0.3 µm, so the right-hand middle band is free
    ax.legend(fontsize=8, loc="lower left" if md.m == 0 else "center right")
    ax.set_title("Derivative continuity at x = ±d in both solvers", fontsize=10)
fig.suptitle("Meep's 1-D eigenmode (MPB) profiles lie on top of the analytic slab solution", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "07_meep_validation.png"); plt.close(fig)
RESULTS["meep"]["runtime_s"] = time.time() - t_meep

# ----------------------------------------------------------------------------- figure 3: thickness sweep (+ Meep points)
thick_sweep = np.linspace(0.05, 0.80, 301)
MAXM = 4
neff_t = np.full((MAXM, thick_sweep.size), np.nan)
dec_t = np.full_like(neff_t, np.nan)
conf_t = np.full_like(neff_t, np.nan)
for i, t in enumerate(thick_sweep):
    for md in st.solve_te_modes(t / 2, LAM, N1, N2):
        if md.m < MAXM:
            neff_t[md.m, i] = md.neff; dec_t[md.m, i] = md.decay_len_um * 1e3; conf_t[md.m, i] = md.confinement
meep_pts = []
for t in (0.15, 0.22, 0.30, 0.40, 0.50, 0.65):
    an = st.solve_te_modes(t / 2, LAM, N1, N2)
    _, mr = meep_modes(t, LAM, N1, N2, min(len(an), MAXM), res=100)
    for a_, r_ in zip(an, mr):
        meep_pts.append({"thickness_nm": t * 1e3, "m": a_.m, "neff_analytic": a_.neff, "neff_meep": r_["neff"],
                         "diff_pct": 100 * (r_["neff"] / a_.neff - 1)})
RESULTS["meep"]["thickness_sweep_points"] = meep_pts
RESULTS["meep"]["thickness_sweep_max_abs_diff_pct"] = max(abs(p["diff_pct"]) for p in meep_pts)
print(f"  Meep vs analytic over the thickness sweep: max |diff| = {RESULTS['meep']['thickness_sweep_max_abs_diff_pct']:.3f} %")

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
ax = axs[0]
for m in range(MAXM):
    ax.plot(thick_sweep * 1e3, neff_t[m], color=SERIES[m], label=f"TE{m} analytic")
for m in range(MAXM):
    pts = [p for p in meep_pts if p["m"] == m]
    ax.plot([p["thickness_nm"] for p in pts], [p["neff_meep"] for p in pts], "x", color=SERIES[m], ms=7, mew=1.8,
            label="Meep" if m == 0 else None)
ax.set_ylim(1.35, 3.64)   # room above the n1 line for its label (the legend sits in the upper-left corner)
ax.axhline(N1, color=PALETTE["muted"], lw=0.8, ls=":"); ax.text(795, N1 + 0.015, "n₁ = 3.50", fontsize=8, color=PALETTE["muted"], ha="right", va="bottom")
ax.axhline(N2, color=PALETTE["muted"], lw=0.8, ls=":"); ax.text(60, N2 + 0.03, "n₂ = 1.45", fontsize=8, color=PALETTE["muted"])
for m in (1, 2, 3):
    tc = st.cutoff_thickness_um(m, LAM, N1, N2) * 1e3
    ax.axvline(tc, color=SERIES[m], lw=0.8, ls="--")
    # label rotated, just left of its own dashed line, starting above the n2 floor: that strip is empty
    # (the TE_m curve rises to the right of the line and the lower modes are far above)
    ax.text(tc - 4, N2 + 0.06, f"TE{m} cutoff {tc:.0f} nm", rotation=90, ha="right", va="bottom", fontsize=7.5, color=SERIES[m])
ax.axvline(THICK * 1e3, color=PALETTE["ink"], lw=1.0); ax.text(THICK * 1e3 + 6, 2.75, "220 nm", fontsize=8)
ax.set_xlabel("core thickness 2d (nm)"); ax.set_ylabel("n_eff = β/k₀")
ax.set_title("n_eff rises from n₂ (cutoff) toward n₁ as the core thickens;\neach new TE_m appears at V = mπ/2 (dashed)", fontsize=9)
ax.legend(fontsize=8)
ax = axs[1]
for m in range(MAXM):
    ax.semilogy(thick_sweep * 1e3, dec_t[m], color=SERIES[m], label=f"TE{m}")
ax.axvline(THICK * 1e3, color=PALETTE["ink"], lw=1.0); ax.text(THICK * 1e3 + 6, 2500, "220 nm", fontsize=8)
ax.axhline(te0.decay_len_um * 1e3, color=SERIES[0], lw=0.8, ls=":")
ax.text(300, 40, f"dotted: TE0 at 220 nm, 1/γ = {te0.decay_len_um*1e3:.0f} nm", fontsize=8, color=SERIES[0])
ax.set_xlabel("core thickness 2d (nm)"); ax.set_ylabel("cladding decay length 1/γ (nm)")
ax.set_title("Near cutoff the tail becomes very long (mode barely bound);\nTE1 at 220 nm has 1/γ ≈ 650 nm", fontsize=9)
ax.set_ylim(30, 5000); ax.legend(fontsize=8)
ax = axs[2]
for m in range(MAXM):
    ax.plot(thick_sweep * 1e3, conf_t[m], color=SERIES[m], label=f"TE{m}")
ax.axvline(THICK * 1e3, color=PALETTE["ink"], lw=1.0); ax.text(THICK * 1e3 - 6, 0.60, "220 nm", fontsize=8, ha="right")
# label at the left, where TE0 is still below 0.8 and no other curve exists (the curves cross 0.85 between 220 and 800 nm)
ax.axhline(REF.confinement, color=PALETTE["muted"], lw=0.8, ls=":")
ax.text(45, REF.confinement + 0.015, "REF strip\nΓ = 0.85", fontsize=8, color=PALETTE["muted"], va="bottom")   # two lines keep it left of 150 nm
ax.set_xlabel("core thickness 2d (nm)"); ax.set_ylabel("Γ_F = fraction of |F|² (power) in the core")
ax.set_title(f"Confinement: TE0 at 220 nm holds Γ_F = {te0.confinement:.2f} of its power in silicon\n(the energy fraction Γ_E = {te0.energy_fraction:.3f} is the thermo-optic weight)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "07_sweep_thickness.png"); plt.close(fig)

# ----------------------------------------------------------------------------- figure 4: wavelength sweep at 220 nm
lam_sweep = np.linspace(1.0, 2.0, 401)
neff_l = np.full((2, lam_sweep.size), np.nan)
for i, l in enumerate(lam_sweep):
    for md in st.solve_te_modes(D, l, N1, N2):
        if md.m < 2:
            neff_l[md.m, i] = md.neff
ng_l = np.full(lam_sweep.size, np.nan)
for i, l in enumerate(lam_sweep):
    ng_l[i] = st.group_index(D, l, N1, N2, 0)
lam_c1 = st.cutoff_wavelength_um(1, D, N1, N2)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axs[0]
ax.plot(lam_sweep * 1e3, neff_l[0], color=SERIES[0], label="TE0 n_eff")
ax.plot(lam_sweep * 1e3, neff_l[1], color=SERIES[1], label="TE1 n_eff")
ax.plot(lam_sweep * 1e3, ng_l, color=SERIES[2], label="TE0 n_g = n_eff − λ dn_eff/dλ (fixed n₁, n₂)")
ax.axhline(N1, color=PALETTE["muted"], lw=0.8, ls=":"); ax.axhline(N2, color=PALETTE["muted"], lw=0.8, ls=":")
ax.axvline(lam_c1 * 1e3, color=SERIES[1], lw=0.8, ls="--"); ax.text(lam_c1 * 1e3 + 8, 2.2, f"TE1 cutoff\nλ = {lam_c1*1e3:.0f} nm", fontsize=8, color=SERIES[1])
ax.axvline(REF.lambda_nm, color=PALETTE["ink"], lw=1.0); ax.text(REF.lambda_nm + 8, 3.3, "1310 nm", fontsize=8)
ax.plot([REF.lambda_nm], [REF.neff], "s", color=SERIES[3], ms=7, label=f"textbook strip n_eff = {REF.neff} (REF)")
ax.plot([REF.lambda_nm], [REF.ng], "D", color=SERIES[3], ms=7, label=f"textbook strip n_g = {REF.ng} (REF)")
ax.set_xlabel("vacuum wavelength λ₀ (nm)"); ax.set_ylabel("index")
ax.set_title("Longer wavelength = weaker confinement: n_eff falls toward n₂; TE1 is cut off above 1402 nm", fontsize=9)
ax.legend(fontsize=7.5, loc="lower right")   # the lower-right quadrant (λ > 1450 nm, index < 2.4) holds no curve
ax = axs[1]
for i, l in enumerate((1.0, 1.31, 1.6, 2.0)):
    mds = st.solve_te_modes(D, l, N1, N2)
    F, _ = st.profile(mds[0], x)
    ax.plot(x * 1e3, F, color=SERIES[i], label=f"λ = {l*1e3:.0f} nm: 1/γ = {mds[0].decay_len_um*1e3:.0f} nm, Γ = {mds[0].confinement:.2f}")
ax.axvspan(-D * 1e3, D * 1e3, color=SERIES[0], alpha=0.10, lw=0)
ax.set_xlabel("x (nm)"); ax.set_ylabel("TE0 F(x) (F(0) = 1)"); ax.legend(fontsize=8)
ax.set_title("The same 220 nm core holds a red mode less tightly: the tail spreads into the cladding", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "07_sweep_wavelength.png"); plt.close(fig)

# ----------------------------------------------------------------------------- figure 5 + video: two plane waves make the mode (notes 21)
h, g, beta = te0.h_per_um, te0.gamma_per_um, te0.beta_per_um
theta = te0.theta_deg
omega = 2 * np.pi * st.C0 / (LAM * 1e-6)          # rad/s
T_fs = 2 * np.pi / omega * 1e15
zz = np.linspace(0, 2 * te0.lambda_g_um, 361)     # two guided wavelengths
xx = np.linspace(-0.40, 0.40, 241)
Z, X = np.meshgrid(zz, xx)
core = np.abs(X) <= D
F2d, _ = st.profile(te0, xx)
F2d = F2d[:, None] * np.ones_like(Z)


def frames_at(t_fs):
    """Notes' convention cos(ωt − k·r): k·r = h x + β z is the wave with k = (+h, β), whose crests
    move toward +x and +z; k·r = −h x + β z is k = (−h, β), crests moving toward −x."""
    wt = omega * t_fs * 1e-15
    w1 = np.where(core, 0.5 * np.cos(wt - beta * Z - h * X), np.nan)   # k = (+h, β): k·r = hx + βz
    w2 = np.where(core, 0.5 * np.cos(wt - beta * Z + h * X), np.nan)   # k = (−h, β): k·r = −hx + βz
    total = F2d * np.cos(wt - beta * Z)
    return w1, w2, total


def _crest_drift_x(wave_fn, dt_fs=0.2):
    """Numerical check of the k_x sign: x of the crest at fixed z = 0 between t = 0 and t = dt (µm)."""
    def crest_x(t):
        col = wave_fn(t)[:, 0]
        return xx[np.nanargmax(col)]
    return float(crest_x(dt_fs) - crest_x(0.0))


def setup_axes(axs, t_fs):
    w1, w2, total = frames_at(t_fs)
    ims = []
    titles = [f"plane wave 1: ½cos(ωt − βz − hx), k = (+h, β), {theta:.1f}° from the x-normal; crests drift toward +x",
              f"plane wave 2: ½cos(ωt − βz + hx), k = (−h, β); crests drift toward −x",
              "sum = F(x)cos(ωt − βz): standing across x, travelling along z"]
    for ax, arr, ttl in zip(axs, (w1, w2, total), titles):
        im = ax.imshow(arr, extent=[zz[0], zz[-1], xx[0], xx[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.axhline(D, color=PALETTE["ink"], lw=0.8); ax.axhline(-D, color=PALETTE["ink"], lw=0.8)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_ylabel("x (µm)")
        ims.append(im)
    # k-vector arrows on the first two panels (drawn from the centre)
    L = 0.22
    for ax, sgn in zip(axs[:2], (+1, -1)):
        dz, dx = L * np.cos(np.radians(90 - theta)), sgn * L * np.sin(np.radians(90 - theta))
        ax.add_patch(FancyArrowPatch((zz[-1] * 0.5, 0), (zz[-1] * 0.5 + dz * zz[-1] / 0.8, dx), arrowstyle="-|>", mutation_scale=14, color=PALETTE["ink"], lw=1.5))
    axs[2].set_xlabel(f"z (µm)   —   λ_g = 2π/β = {te0.lambda_g_um*1e3:.0f} nm")
    return ims


t_video = time.time()
fig, axs = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
ims = setup_axes(axs, 0.0)
sup = fig.suptitle("", fontsize=10)
N_FR = 120
times = np.linspace(0, 2 * T_fs, N_FR, endpoint=False)


def update(i):
    w1, w2, total = frames_at(times[i])
    for im, arr in zip(ims, (w1, w2, total)):
        im.set_data(arr)
    sup.set_text(f"TE0 of the 220 nm slab as two plane waves at ±{theta:.1f}° (critical angle {RESULTS['critical_angle_deg']:.1f}°): "
                 f"t = {times[i]:.2f} fs of T = {T_fs:.2f} fs")
    return ims


fig.tight_layout(rect=(0, 0, 1, 0.96))
anim = FuncAnimation(fig, update, frames=N_FR, blit=False)
anim.save(OUT / "07_plane_wave_decomposition.mp4", writer=FFMpegWriter(fps=30, bitrate=2500))
plt.close(fig)

# contact sheet: 4 instants x 3 panels
fig, axs = plt.subplots(3, 4, figsize=(16, 8), sharex=True, sharey=True)
for c, tf in enumerate(np.linspace(0, 0.75 * T_fs, 4)):
    w1, w2, total = frames_at(tf)
    for r, arr in enumerate((w1, w2, total)):
        ax = axs[r, c]
        ax.imshow(arr, extent=[zz[0], zz[-1], xx[0], xx[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.axhline(D, color=PALETTE["ink"], lw=0.8); ax.axhline(-D, color=PALETTE["ink"], lw=0.8)
        if r == 0: ax.set_title(f"t = {tf:.2f} fs ({tf/T_fs:.2f} T)", fontsize=9)
        if c == 0: ax.set_ylabel(["wave 1, k = (+h, β)", "wave 2, k = (−h, β)", "sum = mode"][r] + "\nx (µm)")
        if r == 2: ax.set_xlabel("z (µm)")
fig.suptitle(f"Frames of 07_plane_wave_decomposition.mp4: the two diagonal waves interfere into a pattern fixed in x that slides along z at v_p = c/{te0.neff:.3f}", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "07_plane_wave_decomposition_frames.png"); plt.close(fig)
# sign check of the labels: in cos(ωt − k·r) a crest at fixed z moves toward +x when k_x > 0
drift1 = _crest_drift_x(lambda t: frames_at(t)[0])
drift2 = _crest_drift_x(lambda t: frames_at(t)[1])
assert drift1 > 0 and drift2 < 0, f"k_x sign labels inconsistent with the drawn fields: {drift1}, {drift2}"
print(f"  plane-wave k_x check: wave 1 crest drifts {drift1*1e3:+.1f} nm in 0.2 fs (k_x = +h), wave 2 {drift2*1e3:+.1f} nm (k_x = −h)")
RESULTS["plane_wave_decomposition"] = {
    "theta_from_normal_deg": theta, "critical_angle_deg": RESULTS["critical_angle_deg"],
    "h_rad_per_um": h, "beta_rad_per_um": beta, "n1_k0_rad_per_um": N1 * K0,
    "check_h2_plus_beta2_equals_n1k0_2": float(np.sqrt(h**2 + beta**2) / (N1 * K0)),
    "wave1_kx_sign": "+h", "wave1_crest_drift_x_nm_per_0p2fs": drift1 * 1e3,
    "wave2_kx_sign": "-h", "wave2_crest_drift_x_nm_per_0p2fs": drift2 * 1e3,
    "period_fs": T_fs, "video_seconds": N_FR / 30,
}
RESULTS["video_runtime_s"] = time.time() - t_video

# ----------------------------------------------------------------------------- figure 6: the physical field E_y(x,z) with the tails
fig, ax = plt.subplots(figsize=(10, 4.2))
zz2 = np.linspace(0, 3 * te0.lambda_g_um, 541)
xx2 = np.linspace(-0.45, 0.45, 301)
Z2, X2 = np.meshgrid(zz2, xx2)
Fp, _ = st.profile(te0, xx2)
field = Fp[:, None] * np.cos(beta * Z2)
im = ax.imshow(field, extent=[zz2[0], zz2[-1], xx2[0], xx2[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.axhline(D, color=PALETTE["ink"], lw=0.8); ax.axhline(-D, color=PALETTE["ink"], lw=0.8)
for k in range(4):
    ax.axvline(k * te0.lambda_g_um, color=PALETTE["ink2"], lw=0.7, ls=":")
ax.annotate("", xy=(te0.lambda_g_um, 0.38), xytext=(0, 0.38), arrowprops=dict(arrowstyle="<->", color=PALETTE["ink"]))
ax.text(te0.lambda_g_um / 2, 0.40, f"λ_g = λ₀/n_eff = {te0.lambda_g_um*1e3:.0f} nm", ha="center", fontsize=8)
ax.text(zz2[-1] * 0.99, D + 0.02, f"tail: e^(−γ(x−d)), 1/γ = {te0.decay_len_um*1e3:.0f} nm", ha="right", fontsize=8, color=PALETTE["ink"])
ax.set_xlabel("z (µm)"); ax.set_ylabel("x (µm)")
ax.set_title(f"Snapshot of E_y(x, z, t=0) = F(x)cos(βz) for TE0: the profile repeats every λ_g = {te0.lambda_g_um*1e3:.0f} nm, shorter than λ₀ = 1310 nm and longer than λ₀/n₁ = {LAM/N1*1e3:.0f} nm", fontsize=9)
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("E_y / F(0)")
fig.tight_layout(); fig.savefig(OUT / "07_field_snapshot.png"); plt.close(fig)

# ----------------------------------------------------------------------------- results + tools
def ffmpeg_version(binary="/opt/homebrew/bin/ffmpeg"):
    """Read the installed ffmpeg version live (first line of `ffmpeg -version`: 'ffmpeg version X ...')."""
    try:
        first = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        return first.split()[2] if first.startswith("ffmpeg version") else first
    except Exception as exc:  # noqa: BLE001 - documentation only; never fail the run over a version string
        return f"unknown ({exc.__class__.__name__})"
FFMPEG_BIN = matplotlib.rcParams.get("animation.ffmpeg_path", "ffmpeg")   # what FFMpegWriter actually calls
if FFMPEG_BIN == "ffmpeg":
    FFMPEG_BIN = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFMPEG_VERSION = ffmpeg_version(FFMPEG_BIN)
RESULTS["ffmpeg"] = {"binary": FFMPEG_BIN, "version": FFMPEG_VERSION}

RESULTS["headline"] = {
    "V_number": V,
    "n_guided_te_modes": len(modes),
    "te0_neff": te0.neff, "te0_neff_notes_sec22": 2.97, "te0_neff_agreement_pct": 100 * te0.neff / 2.97,
    "te0_beta_rad_per_um": te0.beta_per_um,
    "te0_decay_length_nm": te0.decay_len_um * 1e3, "decay_length_notes_sec23_nm": 80.0,
    "te0_decay_length_agreement_pct": 100 * te0.decay_len_um * 1e3 / 80.0,
    "te0_lambda_g_nm": te0.lambda_g_um * 1e3, "lambda_g_notes_sec22_nm": 441.0,
    "te0_lambda_g_agreement_pct": 100 * te0.lambda_g_um * 1e3 / 441.0,
    "te0_theta_deg": te0.theta_deg, "critical_angle_deg": RESULTS["critical_angle_deg"],
    "te0_confinement": te0.confinement, "confinement_REF_strip": REF.confinement,
    "te1_neff": modes[1].neff, "te1_decay_length_nm": modes[1].decay_len_um * 1e3,
    "te1_cutoff_thickness_nm": RESULTS["cutoff"]["te1_thickness_nm_at_1310"],
    "te1_cutoff_wavelength_nm": RESULTS["cutoff"]["te1_wavelength_nm_at_220nm"],
    "te0_ng_slab": ng_te0, "ng_REF_strip": REF.ng,
    "meep_te0_neff": RESULTS["meep"]["modes"][0]["neff_meep"], "meep_te0_neff_agreement_pct": 100 + RESULTS["meep"]["modes"][0]["neff_diff_pct"],
    "meep_te1_neff": RESULTS["meep"]["modes"][1]["neff_meep"], "meep_te1_neff_agreement_pct": 100 + RESULTS["meep"]["modes"][1]["neff_diff_pct"],
    "meep_te0_ng": RESULTS["meep"]["modes"][0]["ng_meep"], "meep_te0_ng_agreement_pct": 100 + RESULTS["meep"]["modes"][0]["ng_diff_pct"],
    "meep_thickness_sweep_max_abs_diff_pct": RESULTS["meep"]["thickness_sweep_max_abs_diff_pct"],
    "te0_energy_fraction_Gamma_E": gamma_E,
    "dneff_dn1_finite_difference": dneff_dn1, "dneff_dn1_identity_ng_Gamma_E_over_n1": ident_ng, "dneff_dn1_identity_n1_Gamma_F_over_neff": ident_neff,
    "dneff_dT_per_K_slab": dneff_dT, "dneff_dT_per_K_textbook_0p85": dneff_dT_textbook,
    "dlambda_dT_pm_per_K_slab": dlam_dT_slab_ng, "dlambda_dT_pm_per_K_slab_closed_form_ng_free": dlam_dT_closed,
    "dlambda_dT_pm_per_K_slab_silicon_term": dlam_dT_si_part, "dlambda_dT_pm_per_K_slab_silica_term": dlam_dT_clad_part,
    "dlambda_dT_pm_per_K_textbook": dlam_dT_textbook, "dlambda_dT_pm_per_K_REF": REF.dlambda_dT_pm_per_K,
    "Gamma_E_needed_for_REF_50_pm_per_K": gamma_E_for_ref_with_clad,
}
RESULTS["runtime_s"] = time.time() - T_START
with open(OUT / "results.json", "w") as fh:
    json.dump(RESULTS, fh, indent=2, default=float)
with open(OUT / "results.txt", "w") as fh:
    fh.write(f"07_slab_modes_graphical  (lambda {REF.lambda_nm} nm, 2d {THICK*1e3:.0f} nm, n1 {N1}, n2 {N2})\n")
    fh.write(f"V = {V:.4f}  ->  {len(modes)} guided TE modes (TE_m needs V > m*pi/2)\n")
    for md in modes:
        fh.write(f"TE{md.m}: n_eff {md.neff:.5f}  beta {md.beta_per_um:.4f} rad/um  1/gamma {md.decay_len_um*1e3:.1f} nm  "
                 f"lambda_g {md.lambda_g_um*1e3:.1f} nm  theta {md.theta_deg:.2f} deg  Gamma {md.confinement:.3f}\n")
    for r in RESULTS["meep"]["modes"]:
        fh.write(f"Meep TE{r['m']}: n_eff {r['neff_meep']:.5f} ({r['neff_diff_pct']:+.3f} %)  n_g {r['ng_meep']:.4f} ({r['ng_diff_pct']:+.3f} %)\n")
    fh.write(f"TE1 cutoff thickness at 1310 nm: {RESULTS['cutoff']['te1_thickness_nm_at_1310']:.1f} nm; TE1 cutoff wavelength at 220 nm: {RESULTS['cutoff']['te1_wavelength_nm_at_220nm']:.1f} nm\n")
    fh.write(f"thermo-optic TE0: Gamma_F {gamma_F:.4f}, Gamma_E {gamma_E:.4f}; dn_eff/dn1 {dneff_dn1:.5f} = (n_g/n1) Gamma_E = (n1/n_eff) Gamma_F; "
             f"dn_eff/dT {dneff_dT:.3e} /K -> dlambda/dT {dlam_dT_slab_ng:.1f} pm/K = lambda[Gamma_E dn1/dT/n1 + (1-Gamma_E) dn2/dT/n2] "
             f"= {dlam_dT_si_part:.1f} + {dlam_dT_clad_part:.1f} pm/K (n_g cancels); REF 50 pm/K needs Gamma_E = {gamma_E_for_ref_with_clad:.3f}\n")
    fh.write(f"runtime {RESULTS['runtime_s']:.1f} s\n")

tools = [
    {"tool": "numpy + scipy.optimize.brentq (slab_te.py)", "version": f"numpy {np.__version__}, scipy {__import__('scipy').__version__}",
     "what_it_is": "NumPy is the array library every Python simulation is built on; SciPy adds numerical routines, here the bracketed root finder brentq (Brent's method, guaranteed convergence when the function changes sign on the bracket).",
     "used_for": "Solving the transcendental TE eigenvalue equations h tan(hd) = γ and −h cot(hd) = γ on the circle (hd)² + (γd)² = V² (notes 19), one root per branch; building F(x), F′(x), H_x, H_z (notes 18); closed-form confinement Γ_F (|F|² fraction) and energy fraction Γ_E (n²|F|² fraction); n_g from a centred finite difference of n_eff(λ) (notes 22); thickness and wavelength sweeps for the cutoffs; the thermo-optic partials ∂n_eff/∂n₁, ∂n_eff/∂n₂ by finite differences of the exact root, checked against the perturbation identities.",
     "result": f"220 nm slab at 1310 nm: V = {V:.3f}, two TE modes. TE0: n_eff = {te0.neff:.4f}, β = {te0.beta_per_um:.3f} rad/µm, 1/γ = {te0.decay_len_um*1e3:.1f} nm (notes 23 quotes 80 nm: {RESULTS['te0']['decay_length_vs_notes_sec23_pct']:+.1f} %), λ_g = {te0.lambda_g_um*1e3:.0f} nm, Γ_F = {te0.confinement:.3f}, Γ_E = {te0.energy_fraction:.3f}. TE1: n_eff = {modes[1].neff:.4f}, 1/γ = {modes[1].decay_len_um*1e3:.0f} nm (barely guided). TE1 cutoff thickness {RESULTS['cutoff']['te1_thickness_nm_at_1310']:.0f} nm, cutoff wavelength {RESULTS['cutoff']['te1_wavelength_nm_at_220nm']:.0f} nm. Thermo-optic: ∂n_eff/∂n₁ = {dneff_dn1:.4f} = (n_g/n₁)Γ_E = (n₁/n_eff)Γ_F to 1e-4, dn_eff/dT = {dneff_dT:.3e} /K, dλ_r/dT = {dlam_dT_slab_ng:.1f} pm/K = λ[Γ_E dn₁/dT/n₁ + (1 − Γ_E) dn₂/dT/n₂] (n_g cancels) vs REF 50 pm/K, which needs Γ_E = {gamma_E_for_ref_with_clad:.2f}.",
     "how_to_observe": "cd experiments/07_slab_modes_graphical && ../../.meep/bin/python run.py ; look at out/07_graphical_construction.png (roots = dots where the circle meets a branch), out/07_mode_profiles.png, out/07_sweep_thickness.png, out/07_sweep_wavelength.png; numbers in out/results.json. Change THICK / LAM / N1 / N2 at the top of run.py, or call slab_te.solve_te_modes(d, lam, n1, n2) directly."},
    {"tool": "Meep (pymeep) get_eigenmode = MPB eigenmode solver", "version": f"meep {mp.__version__}",
     "what_it_is": "Meep is the open-source FDTD electromagnetic simulator; its get_eigenmode routine calls the MPB plane-wave frequency-domain eigensolver on a cross-section of the FDTD cell to find the guided modes and their propagation constants. It is normally used to build mode sources and to decompose FDTD fields into modes.",
     "used_for": "An independent check of β: a 2-D cell holding the 220 nm slab (thickness along Meep y, propagation along Meep x, invariant z; the notes' E_y is Meep's Ez, parity ODD_Z), with get_eigenmode on the 1-D line across the slab for EVEN_Y (TE0) and ODD_Y (TE1); the same at six thicknesses; and a resolution-convergence check.",
     "result": f"TE0 n_eff = {RESULTS['meep']['modes'][0]['neff_meep']:.5f} vs analytic {te0.neff:.5f} ({RESULTS['meep']['modes'][0]['neff_diff_pct']:+.3f} %); TE1 n_eff = {RESULTS['meep']['modes'][1]['neff_meep']:.5f} vs {modes[1].neff:.5f} ({RESULTS['meep']['modes'][1]['neff_diff_pct']:+.3f} %); Meep n_g(TE0) = {RESULTS['meep']['modes'][0]['ng_meep']:.4f} vs finite-difference {ng_te0:.4f} ({RESULTS['meep']['modes'][0]['ng_diff_pct']:+.3f} %); max |Δn_eff| over the thickness sweep {RESULTS['meep']['thickness_sweep_max_abs_diff_pct']:.3f} %. Field profiles overlay the analytic F and F′ (out/07_meep_validation.png).",
     "how_to_observe": "Same run.py (needs the .meep interpreter, hence the .uses_meep marker); out/07_meep_validation.png and the 'meep' block of out/results.json. Change res in meep_modes(...) to see the discretisation error shrink (table te0_resolution_convergence)."},
    {"tool": "matplotlib + FFMpegWriter (ffmpeg)", "version": f"matplotlib {matplotlib.__version__}, ffmpeg {FFMPEG_VERSION} ({FFMPEG_BIN})",
     "what_it_is": "matplotlib is the standard Python plotting library; its animation module renders a FuncAnimation frame by frame and pipes the frames to ffmpeg, which encodes the mp4.",
     "used_for": "All figures, plus the video of the two-plane-wave decomposition of TE0 (notes 21): each panel is an imshow of ½cos(ωt − βz ∓ hx) inside the core (k = (±h, β)) and of the full F(x)cos(ωt − βz) with its evanescent tails, animated over two optical periods.",
     "result": f"out/07_plane_wave_decomposition.mp4 ({N_FR/30:.0f} s, {N_FR} frames) and its contact sheet; the k-vectors of the two waves are tilted ±{theta:.1f}° from the boundary normal (>{RESULTS['critical_angle_deg']:.1f}° critical), √(h² + β²)/(n₁k₀) = {RESULTS['plane_wave_decomposition']['check_h2_plus_beta2_equals_n1k0_2']:.6f} (notes 4), and a crest-tracking check confirms wave 1 drifts toward +x ({drift1*1e3:+.0f} nm per 0.2 fs) and wave 2 toward −x ({drift2*1e3:+.0f} nm), matching the k = (+h, β) / (−h, β) labels.",
     "how_to_observe": "open out/07_plane_wave_decomposition.mp4; watch the top two panels slide diagonally (wave 1 up and to the right, wave 2 down and to the right) while the bottom one keeps a fixed x-shape and slides along z; stills in out/07_plane_wave_decomposition_frames.png. Change N_FR or the time span in run.py."},
    {"tool": "Jupyter notebook (nbformat / nbconvert, ipywidgets)", "version": "see explore.ipynb metadata (kernel photonics-sims)",
     "what_it_is": "Jupyter notebooks mix code, output and text; ipywidgets adds sliders that re-run a function; nbconvert executes a notebook headlessly and stores the outputs.",
     "used_for": "explore.ipynb (generated by build_notebook.py with nbformat): sliders for thickness (50–800 nm), wavelength (1000–2000 nm) and the two indices that redraw the graphical construction, the mode list and the profiles; a cutoff map N_TE(2d, λ); and an n_eff-vs-n_g cell that recomputes the ring FSR with the slab n_g and with REF n_g = 4.2 (uses slab_te.py, no Meep).",
     "result": "Executed with the default (220 nm, 1310 nm) values so the saved outputs show the same numbers as run.py: 2 TE modes, TE0 n_eff 2.988, slab n_g 3.63 -> FSR 11.9 nm versus 10.3 nm with n_g 4.2.",
     "how_to_observe": "cd experiments/07_slab_modes_graphical && ../../.venv/bin/jupyter lab explore.ipynb ; drag the sliders. Rebuild with ../../.venv/bin/python build_notebook.py and re-execute headlessly with ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb"},
]
with open(OUT / "tools.json", "w") as fh:
    json.dump(tools, fh, indent=2)
print(f"done in {RESULTS['runtime_s']:.1f} s; outputs in {OUT}")
