"""04_sellmeier_dispersion: from the Sellmeier fit to group index and material dispersion.

Notes §9 (Sellmeier form), §10 (UV vs IR curvature), §11 (n_g = n − λ dn/dλ, D = −(λ/c) d²n/dλ²).

Run:  cd experiments/04_sellmeier_dispersion && ../../.venv/bin/python run.py
Regenerates everything in out/.
"""
import sys, os, pathlib, json, time, platform, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE, C0
from common.units import delta_f_ghz_from_delta_lambda_nm, delta_lambda_nm_from_delta_f_ghz
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

# Regenerate from scratch: remove every previous output so nothing stale survives a re-run.
for _p in OUT.iterdir():
    if _p.is_file():
        _p.unlink()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import scipy, scipy.optimize
import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sellmeier as sm
from sellmeier import lam, c_sym, B1, B2, B3, C1, C2, C3

T0 = time.time()
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)


# ==========================================================================
# 1. Symbolic derivation (sympy)                                     notes §11, §10
# ==========================================================================
say("=" * 78)
say("1. SYMBOLIC DERIVATION (sympy)")
say("=" * 78)

omega = sp.symbols("omega", positive=True)
nf = sp.Function("n")                       # n as an unknown function

# (a) group index from k(ω) = n ω / c, notes §11
lam_of_omega = 2 * sp.pi * c_sym / omega
k = nf(lam_of_omega) * omega / c_sym
dk_domega = sp.diff(k, omega)
ng_from_omega = sp.simplify(c_sym * dk_domega)              # c dk/dω, still in terms of ω
ng_in_lambda = sp.simplify(ng_from_omega.subs(omega, 2 * sp.pi * c_sym / lam))  # back to λ
say("\n(a) k(ω) = n(λ(ω)) ω / c with λ = 2πc/ω.  n_g ≡ c dk/dω =")
say(sp.pretty(ng_in_lambda, use_unicode=True))
target_ng = nf(lam) - lam * sp.Derivative(nf(lam), lam)
assert sp.simplify(ng_in_lambda - target_ng.doit()) == 0
say("   which is exactly  n_g = n − λ dn/dλ   (checked: difference simplifies to 0)")

# (b) dispersion parameter from group delay per length τ_g/L = n_g/c
tau_per_L = target_ng / c_sym
D_expr_generic = sp.simplify(sp.diff(tau_per_L, lam))
say("\n(b) D = d(τ_g/L)/dλ = d(n_g/c)/dλ =")
say(sp.pretty(D_expr_generic, use_unicode=True))
target_D = -(lam / c_sym) * sp.Derivative(nf(lam), (lam, 2))
assert sp.simplify(D_expr_generic - target_D.doit()) == 0
say("   which is exactly  D = −(λ/c) d²n/dλ²   (the dn/dλ terms cancel; only curvature survives)")

# (c) chain rule through n = sqrt(1 + S): dn/dλ = S'/(2n), d²n/dλ² = S''/(2n) − S'²/(4n³)  (notes §10)
S = sp.Function("S")(lam)
n_of_S = sp.sqrt(1 + S)
dn_S = sp.simplify(sp.diff(n_of_S, lam))
d2n_S = sp.simplify(sp.diff(n_of_S, lam, 2))
say("\n(c) With n = √(1+S), S = Σ B_i λ²/(λ² − C_i):")
say("    dn/dλ   =", sp.pretty(dn_S, use_unicode=True).replace("\n", "\n              "))
say("    d²n/dλ² =", sp.pretty(d2n_S, use_unicode=True).replace("\n", "\n              "))
Sd, Sdd, nS = sp.symbols("S' S'' n", positive=True)
d2n_sub = d2n_S.subs(sp.Derivative(S, (lam, 2)), Sdd).subs(sp.Derivative(S, lam), Sd).subs(S, nS**2 - 1)
check = sp.simplify(d2n_sub - (Sdd / (2 * nS) - Sd**2 / (4 * nS**3)))
say("    matches S''/(2n) − S'²/(4n³) after S → n² − 1:", check == 0)
assert check == 0

# (d) generic 3-term Sellmeier: the explicit n_g and d²n/dλ²
n_generic = sp.sqrt(sm.sellmeier_n2_generic())
_, dn_g, ng_g, d2n_g, D_g = sm.dispersion_expressions(n_generic)
say("\n(d) Generic Sellmeier n(λ) = √(1 + Σ B_i λ²/(λ² − C_i)):")
say("    n_g(λ) =")
say(sp.pretty(sp.simplify(ng_g), use_unicode=True))

# (e) UV vs IR asymptotics of a single term, notes §10
Bu, lu, Bir, lir = sp.symbols("B_u lambda_u B_ir lambda_ir", positive=True)
S_u = Bu * lam**2 / (lam**2 - lu**2)
S_ir = Bir * lam**2 / (lam**2 - lir**2)
S_u_series = sp.series(S_u, lu, 0, 5).removeO()          # λ ≫ λ_u
S_ir_series = sp.series(S_ir, lam, 0, 6).removeO()       # λ ≪ λ_ir
say("\n(e) One UV term for λ ≫ λ_u:   S_u ≈", sp.pretty(sp.expand(S_u_series), use_unicode=True))
say("        d²S_u/dλ² ≈", sp.pretty(sp.expand(sp.diff(S_u_series, lam, 2)), use_unicode=True), " > 0  (bends UP)")
say("    One IR term for λ ≪ λ_ir:  S_ir ≈", sp.pretty(sp.expand(S_ir_series), use_unicode=True))
say("        d²S_ir/dλ² ≈", sp.pretty(sp.expand(sp.diff(S_ir_series, lam, 2)), use_unicode=True), " < 0  (bends DOWN)")
say("    Both slopes are negative (n falls with λ everywhere in the window); only the CURVATURES have opposite sign,")
say("    so the zero-dispersion wavelength is where the curvatures cancel, not the slopes.")

(OUT / "derivation.txt").write_text("\n".join(LOG) + "\n")

# Equations rendered to PNG with mathtext (no LaTeX on this machine)
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axis("off")
eqs = [
    r"Sellmeier (notes §9):  $n^2(\lambda) - 1 = \sum_i \dfrac{B_i\,\lambda^2}{\lambda^2 - C_i}$,   $\sqrt{C_i}$ = fitted resonance wavelength",
    r"Group index (notes §11):  $n_g = c\,\dfrac{dk}{d\omega} = n + \omega\dfrac{dn}{d\omega} = n - \lambda\dfrac{dn}{d\lambda}$",
    r"Material dispersion (notes §11):  $D = \dfrac{d}{d\lambda}\left(\dfrac{n_g}{c}\right) = -\dfrac{\lambda}{c}\,\dfrac{d^2 n}{d\lambda^2}$   [ps/(nm·km)]",
    r"Chain rule (notes §10):  $\dfrac{dn}{d\lambda} = \dfrac{S'}{2n}$,   $\dfrac{d^2n}{d\lambda^2} = \dfrac{S''}{2n} - \dfrac{S'^2}{4n^3}$,   $S = n^2 - 1$",
    r"UV term ($\lambda \gg \lambda_u$):  $S_u \approx B_u + B_u\lambda_u^2/\lambda^2$,  $S_u'' > 0$;    IR term ($\lambda \ll \lambda_{ir}$):  $S_{ir} \approx -B_{ir}\lambda^2/\lambda_{ir}^2$,  $S_{ir}'' < 0$",
    r"Ring FSR (capstone):  $\Delta f_{FSR} = \dfrac{c}{n_g L}$,   $\Delta\lambda_{FSR} = \dfrac{\lambda^2}{n_g L}$,   small-change rule  $|\Delta f| = \dfrac{c}{\lambda^2}|\Delta\lambda|$",
]
for i, e in enumerate(eqs):
    ax.text(0.01, 0.95 - i * 0.185, e, fontsize=11.5, va="top", ha="left", transform=ax.transAxes)
ax.set_title("The equations this experiment evaluates (all derived symbolically in run.py, see out/derivation.txt)")
fig.savefig(OUT / "equations.png", bbox_inches="tight"); plt.close(fig)

# ==========================================================================
# 2. Numeric evaluation: silica and silicon                          notes §9-§11
# ==========================================================================
say("\n" + "=" * 78)
say("2. NUMERIC EVALUATION")
say("=" * 78)
silica = sm.silica()
silicon = sm.silicon()

lam_sio2 = np.linspace(0.60, 2.00, 1401)
lam_si = np.linspace(1.20, 2.00, 801)          # Salzberg-Villa pole at 1.107 µm; below ~1.1 µm silicon absorbs
L0 = REF.lambda_nm * 1e-3                       # 1.31 µm

res = {}
# --- finite-difference cross-check of the symbolic derivatives -------------
h = 1e-4
fd_dn = (silica.n(lam_sio2 + h) - silica.n(lam_sio2 - h)) / (2 * h)
fd_d2n = (silica.n(lam_sio2 + h) - 2 * silica.n(lam_sio2) + silica.n(lam_sio2 - h)) / h**2
ng_fd = silica.n(lam_sio2) - lam_sio2 * fd_dn
D_fd = sm.D_ps_nm_km(lam_sio2, fd_d2n)
res["fd_check_ng_max_abs_diff"] = float(np.max(np.abs(ng_fd - silica.ng(lam_sio2))))
res["fd_check_D_max_abs_diff_ps_nm_km"] = float(np.max(np.abs(D_fd - silica.D_ps_nm_km(lam_sio2))))
say(f"Finite-difference cross-check (silica, h = {h} µm): max|Δn_g| = {res['fd_check_ng_max_abs_diff']:.2e}, "
    f"max|ΔD| = {res['fd_check_D_max_abs_diff_ps_nm_km']:.2e} ps/(nm·km)")

# --- headline numbers at 1310 and 1550 -------------------------------------
for name, mat in (("silica", silica), ("silicon", silicon)):
    for lam_nm in (1310, 1550):
        l = lam_nm * 1e-3
        res[f"n_{name}_{lam_nm}"] = float(mat.n(l))
        res[f"ng_{name}_{lam_nm}"] = float(mat.ng(l))
        res[f"dn_dlam_{name}_{lam_nm}_per_um"] = float(mat.dn_dlam(l))
        res[f"D_{name}_{lam_nm}_ps_nm_km"] = float(mat.D_ps_nm_km(l))
        res[f"beta2_{name}_{lam_nm}_ps2_km"] = float(mat.beta2_ps2_km(l))
        say(f"{name:8s} λ = {lam_nm} nm: n = {mat.n(l):.5f}, dn/dλ = {mat.dn_dlam(l):.5f} /µm, n_g = {mat.ng(l):.5f}, "
            f"D = {mat.D_ps_nm_km(l):+.3f} ps/(nm·km), β₂ = {mat.beta2_ps2_km(l):+.3f} ps²/km")

# --- zero-dispersion wavelength of silica (root of d²n/dλ² on the full n) ---
zdw = scipy.optimize.brentq(lambda x: float(silica.d2n_dlam2(x)), 1.0, 1.6, xtol=1e-10)
zdw_fd = scipy.optimize.brentq(lambda x: (silica.n(x + h) - 2 * silica.n(x) + silica.n(x - h)) / h**2, 1.0, 1.6)
res["zdw_silica_um"] = float(zdw)
res["zdw_silica_um_finite_difference"] = float(zdw_fd)
# dispersion slope at the ZDW, S0 = dD/dλ in ps/(nm²·km)
S0 = (silica.D_ps_nm_km(zdw + 1e-3) - silica.D_ps_nm_km(zdw - 1e-3)) / (2 * 1e-3) * 1e-3   # per nm
res["dispersion_slope_at_zdw_ps_nm2_km"] = float(S0)
say(f"Zero-dispersion wavelength of bulk silica: {zdw*1e3:.2f} nm (finite-difference root {zdw_fd*1e3:.2f} nm); "
    f"slope S0 = {S0:.4f} ps/(nm²·km)")
# Independent expectation for D(1310): a frequency-domain route that never differentiates n with respect to λ.
# β(ω) = n(λ(ω))·ω/c, β₂ = d²β/dω² by central differences in ω, then D = −(2πc/λ²)·β₂ (notes §12).
# This shares only the Sellmeier table with the λ-route above, so it is a genuine cross-derivation of the same number.
def D_omega_route_ps_nm_km(mat, lam_um, rel_h=1e-3):
    lam_m = lam_um * 1e-6
    w0 = 2 * np.pi * C0 / lam_m
    hw = rel_h * w0
    beta = lambda w: float(mat.n(2 * np.pi * C0 / w * 1e6)) * w / C0      # 1/m
    beta2 = (beta(w0 + hw) - 2 * beta(w0) + beta(w0 - hw)) / hw**2        # s²/m
    return float(-(2 * np.pi * C0 / lam_m**2) * beta2 * 1e6)                # s/m² → ps/(nm·km)


res["D_silica_1310_omega_route_ps_nm_km"] = D_omega_route_ps_nm_km(silica, 1.31)
res["D_silica_1550_omega_route_ps_nm_km"] = D_omega_route_ps_nm_km(silica, 1.55)
say(f"D(1310) by the independent ω-route (β₂ = d²β/dω² by finite differences, D = −2πc β₂/λ²): "
    f"{res['D_silica_1310_omega_route_ps_nm_km']:+.4f} ps/(nm·km) vs λ-route {res['D_silica_1310_ps_nm_km']:+.4f}; "
    f"D(1550): {res['D_silica_1550_omega_route_ps_nm_km']:+.4f} vs {res['D_silica_1550_ps_nm_km']:+.4f}")
# Cross-checks (NOT the headline expectation): Taylor expansions of D(λ) about the zero crossing.
# D(λ) is curved (its slope falls with λ), so the straight line S0·(1310 − ZDW) over 37 nm is a few % high;
# adding the second-order term ½·S0'·(Δλ)² closes most of that gap.
dlam_zdw_nm = 1310.0 - zdw * 1e3
S0_prime = (silica.D_ps_nm_km(zdw + 1e-3) - 2 * silica.D_ps_nm_km(zdw) + silica.D_ps_nm_km(zdw - 1e-3)) / (1e-3)**2 * 1e-6  # ps/(nm³·km)
res["dispersion_curvature_at_zdw_ps_nm3_km"] = float(S0_prime)
res["D_silica_1310_linear_estimate_ps_nm_km"] = float(S0 * dlam_zdw_nm)
res["D_silica_1310_taylor2_estimate_ps_nm_km"] = float(S0 * dlam_zdw_nm + 0.5 * S0_prime * dlam_zdw_nm**2)
res["D_silica_1310_linearisation_error_percent"] = float(100 * (res["D_silica_1310_linear_estimate_ps_nm_km"] / res["D_silica_1310_ps_nm_km"] - 1))
say(f"D(1310) Taylor cross-checks from the ZDW: linear S0·Δλ = {S0:.4f} × {dlam_zdw_nm:.2f} nm = "
    f"{res['D_silica_1310_linear_estimate_ps_nm_km']:+.3f} ps/(nm·km) ({res['D_silica_1310_linearisation_error_percent']:+.1f} % vs exact: the linearisation error); "
    f"second order with S0' = {S0_prime:+.2e} ps/(nm³·km): {res['D_silica_1310_taylor2_estimate_ps_nm_km']:+.3f}")

# Where does the ZDW come from? split d²n/dλ² into UV and IR contributions of S''/(2n) plus the −S'²/(4n³) term
terms = sm.silica_terms(lam_sio2)
n_s = silica.n(lam_sio2)
S1_tot = sum(t[2] for t in terms)
S2_uv = terms[0][3] + terms[1][3]
S2_ir = terms[2][3]
curv_uv = S2_uv / (2 * n_s)
curv_ir = S2_ir / (2 * n_s)
curv_sq = -S1_tot**2 / (4 * n_s**3)
curv_tot = curv_uv + curv_ir + curv_sq
assert np.max(np.abs(curv_tot - silica.d2n_dlam2(lam_sio2))) < 1e-9
i0 = np.argmin(np.abs(lam_sio2 - zdw))
res["zdw_decomposition_per_um2"] = {"uv_terms_S''/2n": float(curv_uv[i0]), "ir_term_S''/2n": float(curv_ir[i0]),
                                    "-S'^2/4n^3": float(curv_sq[i0]), "total": float(curv_tot[i0])}
say(f"At the ZDW, d²n/dλ² contributions (1/µm²): UV terms {curv_uv[i0]:+.5f}, IR term {curv_ir[i0]:+.5f}, "
    f"−S'²/(4n³) {curv_sq[i0]:+.5f}  → total {curv_tot[i0]:+.1e}")

# --- small-change rule: 1 nm ↔ 174.7 GHz at 1310 nm ------------------------
scr = sm.small_change_rule_ghz_per_nm(REF.lambda_nm)
f_exact = C0 / (REF.lambda_nm * 1e-9) - C0 / ((REF.lambda_nm + 1.0) * 1e-9)
res["small_change_ghz_per_nm_1310"] = float(scr)
res["small_change_exact_1nm_step_ghz_1310"] = float(f_exact / 1e9)
res["small_change_common_units_helper_ghz"] = float(delta_f_ghz_from_delta_lambda_nm(1.0, REF.lambda_nm))
res["small_change_ghz_per_nm_1550"] = float(sm.small_change_rule_ghz_per_nm(1550))
say(f"Small-change rule at 1310 nm: c/λ² = {scr:.2f} GHz/nm (exact 1 nm step: {f_exact/1e9:.2f} GHz; "
    f"common.units helper: {res['small_change_common_units_helper_ghz']:.2f} GHz); at 1550 nm: {res['small_change_ghz_per_nm_1550']:.2f} GHz/nm")

# capstone conversions with the rule
res["capstone_conversions"] = {
    "fsr_10.3nm_in_thz": float(delta_f_ghz_from_delta_lambda_nm(REF.fsr_nm, REF.lambda_nm) / 1e3),
    "thermal_50pm_per_K_in_ghz_per_K": float(delta_f_ghz_from_delta_lambda_nm(REF.dlambda_dT_pm_per_K * 1e-3, REF.lambda_nm)),
    "fwhm_374pm_in_ghz": float(delta_f_ghz_from_delta_lambda_nm(REF.fwhm_pm * 1e-3, REF.lambda_nm)),
    "channel_spacing_200ghz_in_nm": float(delta_lambda_nm_from_delta_f_ghz(REF.channel_spacing_ghz, REF.lambda_nm)),
    "delta_opt_108pm_in_ghz": float(delta_f_ghz_from_delta_lambda_nm(REF.delta_opt_pm * 1e-3, REF.lambda_nm)),
    "one_pm_in_mhz": float(delta_f_ghz_from_delta_lambda_nm(1e-3, REF.lambda_nm) * 1e3),
}
for k_, v_ in res["capstone_conversions"].items():
    say(f"   {k_}: {v_:.4g}")

# --- ring FSR: group index, not effective index ----------------------------
L_rt = REF.round_trip_um * 1e-6
fsr_from = {}
for label, idx in (("n_eff textbook 2.5", REF.neff), ("n_eff solver ~2.7", 2.7),
                   ("bulk-Si n_g (this run)", float(silicon.ng(L0))), ("waveguide n_g 4.2 (REF)", REF.ng)):
    f_thz = C0 / (idx * L_rt) / 1e12
    l_nm = (REF.lambda_nm * 1e-9) ** 2 / (idx * L_rt) * 1e9
    fsr_from[label] = {"index": idx, "fsr_thz": float(f_thz), "fsr_nm": float(l_nm)}
    say(f"FSR with {label:26s} ({idx:.3f}): {f_thz:.3f} THz = {l_nm:.2f} nm")
res["fsr_by_index"] = fsr_from
res["fsr_thz_ng42"] = fsr_from["waveguide n_g 4.2 (REF)"]["fsr_thz"]
res["fsr_nm_ng42"] = fsr_from["waveguide n_g 4.2 (REF)"]["fsr_nm"]
# the waveguide contribution to n_g: waveguide n_g (4.2) − bulk silicon n_g
res["ng_waveguide_minus_bulk_si"] = float(REF.ng - silicon.ng(L0))
res["ng_minus_n_bulk_si_1310"] = float(silicon.ng(L0) - silicon.n(L0))
# thermal tuning of a ring: dλ_r/dT = (λ/n_g) Γ dn_si/dT (the group index appears here too)
dl_dT = REF.lambda_nm * 1e3 / REF.ng * (REF.confinement * REF.dn_si_dT + (1 - REF.confinement) * REF.dn_sio2_dT)
res["dlambda_dT_pm_per_K_from_ng"] = float(dl_dT)
res["dlambda_dT_pm_per_K_if_neff_used"] = float(REF.lambda_nm * 1e3 / REF.neff * (REF.confinement * REF.dn_si_dT + (1 - REF.confinement) * REF.dn_sio2_dT))
say(f"Thermal shift dλ_r/dT = (λ/n_g)·(Γ dn_si/dT + (1−Γ) dn_sio2/dT) = {dl_dT:.1f} pm/K "
    f"(REF says {REF.dlambda_dT_pm_per_K}); with n_eff in the denominator it would wrongly be {res['dlambda_dT_pm_per_K_if_neff_used']:.1f} pm/K")

# --- fibre-link relevance: pulse spread over 2 km (material dispersion only) -
# Two spectral-width conventions for the 53.125 Gbaud NRZ signal, both converted to Δλ with the small-change rule:
#   (a) "delay spread across the signal band": the two-sided width 2 × 0.75 × baud = 79.7 GHz (the ±f_3dB of the
#       0.75×baud electro-optic filters used in experiment 05). This is the convention 05 and the brief use (~27 ps at 1550).
#   (b) crude one-sided estimate Δλ ↔ baud rate (first spectral null of NRZ): 53.1 GHz.
F_BAND_GHZ = 2 * 0.75 * REF.baud / 1e9
res["signal_band_two_sided_ghz"] = float(F_BAND_GHZ)
for lam_nm in (1310, 1550):
    D = res[f"D_silica_{lam_nm}_ps_nm_km"]
    ghz_per_nm = sm.small_change_rule_ghz_per_nm(lam_nm)
    dlam_band_nm = F_BAND_GHZ / ghz_per_nm
    dlam_nm = REF.baud / 1e9 / ghz_per_nm
    res[f"spread_2km_{lam_nm}_ps"] = float(abs(D) * 2.0 * dlam_band_nm)                 # (a) headline, matches 05
    res[f"spread_2km_{lam_nm}_ps_one_sided_baud"] = float(abs(D) * 2.0 * dlam_nm)       # (b) crude
    say(f"Over 2 km at {lam_nm} nm: |D|·L·Δλ with Δλ ↔ ±0.75×baud = {F_BAND_GHZ:.1f} GHz two-sided ↔ {dlam_band_nm:.3f} nm → "
        f"{res[f'spread_2km_{lam_nm}_ps']:.2f} ps; crude one-sided Δλ ↔ {REF.baud/1e9:.1f} GHz ↔ {dlam_nm:.3f} nm → "
        f"{res[f'spread_2km_{lam_nm}_ps_one_sided_baud']:.1f} ps (bit period {1e12/REF.baud:.1f} ps)")
res["bit_period_ps"] = 1e12 / REF.baud

# ==========================================================================
# 3. Figures
# ==========================================================================
say("\n" + "=" * 78)
say("3. FIGURES")
say("=" * 78)

# --- Fig A: n and n_g vs λ for both materials ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axs[0]
ax.plot(lam_sio2 * 1e3, silica.n(lam_sio2), color=SERIES[0], label="phase index n")
ax.plot(lam_sio2 * 1e3, silica.ng(lam_sio2), color=SERIES[1], label="group index n_g = n − λ dn/dλ")
ax.axvline(1310, color=PALETTE["muted"], lw=1, ls="--", label="1310 nm (O-band, capstone)")
ax.axvline(1550, color=PALETTE["muted"], lw=1, ls=":", label="1550 nm (C-band)")
ax.axvline(zdw * 1e3, color=SERIES[3], lw=1, ls="-.", label=f"zero-dispersion λ = {zdw*1e3:.0f} nm (n_g minimum)")
ax.annotate(f"n(1310) = {res['n_silica_1310']:.4f}\nn_g(1310) = {res['ng_silica_1310']:.4f}",
            xy=(1310, res['n_silica_1310']), xytext=(1400, 1.452), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
ax.set_xlabel("vacuum wavelength λ₀ (nm)"); ax.set_ylabel("index (dimensionless)")
ax.set_title("Fused silica: n_g sits above n; its minimum is the zero-dispersion λ", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")
ax = axs[1]
ax.plot(lam_si * 1e3, silicon.n(lam_si), color=SERIES[0], label="phase index n")
ax.plot(lam_si * 1e3, silicon.ng(lam_si), color=SERIES[1], label="group index n_g (bulk)")
ax.axhline(REF.ng, color=SERIES[3], lw=1.2, ls="--", label=f"REF waveguide n_g = {REF.ng} (bulk + waveguide dispersion)")
ax.axvline(1310, color=PALETTE["muted"], lw=1, ls="--", label="1310 nm (O-band, capstone)")
ax.annotate(f"n(1310) = {res['n_silicon_1310']:.3f}\nn_g(1310) = {res['ng_silicon_1310']:.3f}",
            xy=(1310, res['n_silicon_1310']), xytext=(1450, 3.95), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
# The textbook n_eff = 2.5 would need a 2.3-4.5 axis that squashes the silicon curves into 15 % of the panel;
# it is quoted as text here and compared numerically in capstone_fsr.png instead.
ax.text(0.98, 0.38, f"REF n_eff = {REF.neff} (textbook; solvers ≈ 2.7)\nlies below this axis; the FSR it\n"
        "would give is compared in capstone_fsr.png", transform=ax.transAxes, fontsize=8,
        va="center", ha="right", color=PALETTE["ink2"])
ax.set_xlabel("vacuum wavelength λ₀ (nm)"); ax.set_ylabel("index (dimensionless)")
ax.set_title("Silicon: bulk n_g = n + 0.18; the strip's n_g = 4.2 is waveguide dispersion", fontsize=9.5)
ax.set_ylim(3.40, 4.45); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "n_and_ng.png"); plt.close(fig)

# --- Fig B: tangent-intercept construction of n_g ---------------------------
def tangent_fig(mat, lam_win, lam_t, fname, ylim):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    lam_full = np.linspace(0.0, lam_win[1], 400)
    ax.plot(lam_win_arr := np.linspace(*lam_win, 600), mat.n(lam_win_arr), color=SERIES[0], label="n(λ)")
    nt, st = float(mat.n(lam_t)), float(mat.dn_dlam(lam_t))
    ax.plot(lam_full, nt + st * (lam_full - lam_t), color=SERIES[1], lw=1.4, ls="--",
            label=f"tangent at λ = {lam_t*1e3:.0f} nm, slope dn/dλ = {st:.4f} /µm")
    ax.plot([lam_t], [nt], "o", color=SERIES[0], ms=6)
    ax.plot([0], [nt - st * lam_t], "s", color=SERIES[1], ms=7, label=f"intercept at λ = 0: n − λ dn/dλ = n_g = {nt - st*lam_t:.4f}")
    ax.plot(lam_win_arr, mat.ng(lam_win_arr), color=SERIES[2], lw=1.4, label="n_g(λ) (traced by the intercept)")
    ax.axhline(nt - st * lam_t, color=PALETTE["line"], lw=1)
    ax.set_xlim(-0.05, lam_win[1] + 0.05); ax.set_ylim(*ylim)
    ax.set_xlabel("vacuum wavelength λ₀ (µm)"); ax.set_ylabel("index (dimensionless)")
    ax.set_title(f"{mat.name.split(' (')[0]}: n_g is where the tangent to n(λ) hits the λ = 0 axis", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)

tangent_fig(silica, (0.6, 2.0), L0, "tangent_intercept_silica.png", (1.42, 1.50))
tangent_fig(silicon, (1.2, 2.0), L0, "tangent_intercept_silicon.png", (3.3, 3.9))

# --- Fig C: D vs λ --------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axs[0]
Dsi = silica.D_ps_nm_km(lam_sio2)
ax.plot(lam_sio2 * 1e3, Dsi, color=SERIES[0], label="D = −(λ/c) d²n/dλ²  (bulk silica, material only)")
ax.axhline(0, color=PALETTE["ink2"], lw=0.8)
ax.axvline(zdw * 1e3, color=SERIES[3], ls="-.", lw=1, label=f"zero-dispersion λ = {zdw*1e3:.1f} nm")
# D(1310) is labelled below the curve, D(1550) above it, so the two labels and their arrows never cross
# and neither sits on the SMF-28 dotted line at 17 ps/(nm·km).
for lam_nm, c_, xytext in ((1310, SERIES[1], (1310, -60)), (1550, SERIES[4], (1230, 48))):
    Dv = res[f"D_silica_{lam_nm}_ps_nm_km"]
    ax.plot([lam_nm], [Dv], "o", color=c_, ms=7)
    ax.annotate(f"D({lam_nm}) = {Dv:+.1f} ps/(nm·km)", xy=(lam_nm, Dv), xytext=xytext, fontsize=8.5,
                ha="center", va="center", arrowprops=dict(arrowstyle="->", color=c_))
ax.axhline(17, color=PALETTE["muted"], ls=":", lw=1, label="SMF-28 spec at 1550 (≈17; includes waveguide dispersion)")
ax.set_ylim(-250, 60)
ax.set_xlabel("vacuum wavelength λ₀ (nm)"); ax.set_ylabel("D  (ps / (nm · km))")
ax.set_title("Silica: D crosses zero at 1.27 µm; the O-band sits on the zero, the C-band does not", fontsize=9.5)
ax.legend(fontsize=7.5, loc="lower right")
ax = axs[1]
ax.plot(lam_si * 1e3, silicon.D_ps_nm_km(lam_si), color=SERIES[0], label="D bulk silicon")
ax.axhline(0, color=PALETTE["ink2"], lw=0.8); ax.axvline(1310, color=PALETTE["muted"], ls="--", lw=1)
ax.annotate(f"D(1310) = {res['D_silicon_1310_ps_nm_km']:+.0f} ps/(nm·km)", xy=(1310, res['D_silicon_1310_ps_nm_km']),
            xytext=(1450, -1500), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
ax.set_xlabel("vacuum wavelength λ₀ (nm)"); ax.set_ylabel("D  (ps / (nm · km))")
ax.set_title("Silicon: D is ~100× silica and negative, but a ring is only 40 µm long", fontsize=9.5)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "dispersion_D.png"); plt.close(fig)

# --- Fig C2: the Sellmeier poles on a log-wavelength axis (notes §8, §9) ----
# Why do resonances at 68 nm, 116 nm and 9.9 µm still set n at 1310 nm?  Because each term
# S_i = B_i λ²/(λ² − C_i) is a pole at √C_i whose tails extend across the whole transparent window.
lam_log = np.logspace(np.log10(0.03), np.log10(30.0), 6000)
fig, ax = plt.subplots(figsize=(10, 4.6))
S_tot = np.zeros_like(lam_log)
for i, ((B, C), c_) in enumerate(zip(zip(sm.SILICA_B, sm.SILICA_C), SERIES)):
    S_i = B * lam_log**2 / (lam_log**2 - C)
    S_tot += S_i
    ax.plot(lam_log * 1e3, np.clip(S_i, -4, 6), color=c_, lw=1.4, ls="--",
            label=f"term {i+1}: B = {B:.3f}, √C = {np.sqrt(C)*1e3:.0f} nm")
    ax.axvline(np.sqrt(C) * 1e3, color=c_, lw=0.8, alpha=0.5)
ax.plot(lam_log * 1e3, np.clip(S_tot, -4, 6), color=PALETTE["ink"], lw=2.2, label="sum  S = n² − 1")
ax.axvspan(600, 2000, color=SERIES[5], alpha=0.12, label="window plotted elsewhere (0.6–2.0 µm)")
ax.axvline(1310, color=SERIES[4], lw=1.4, ls="-.", label=f"1310 nm: n² − 1 = {res['n_silica_1310']**2 - 1:.4f}")
ax.axhline(0, color=PALETTE["ink2"], lw=0.8)
ax.set_xscale("log"); ax.set_ylim(-4, 6); ax.set_xlim(30, 30000)
ax.set_xlabel("vacuum wavelength λ₀ (nm, log axis)"); ax.set_ylabel("S(λ) = n² − 1  (clipped to ±4…6)")
ax.set_title("Silica's Sellmeier fit is three resonances: 1310 nm sits in the gap between the UV poles (68, 116 nm) and the IR pole (9.9 µm)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper center", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "sellmeier_resonances.png"); plt.close(fig)

# --- Fig C3: why the O-band — 53 Gbaud over 2 km, spread vs wavelength -------
lam_ob = np.linspace(1.20, 1.65, 451)
ghz_per_nm_ob = sm.small_change_rule_ghz_per_nm(lam_ob * 1e3)
dlam_band_ob = F_BAND_GHZ / ghz_per_nm_ob                              # nm, two-sided ±0.75×baud band (as in 05)
dlam_ob = REF.baud / 1e9 / ghz_per_nm_ob                               # nm, crude one-sided Δλ ↔ baud
spread_ob = np.abs(silica.D_ps_nm_km(lam_ob)) * 2.0 * dlam_band_ob     # ps over 2 km
spread_ob_crude = np.abs(silica.D_ps_nm_km(lam_ob)) * 2.0 * dlam_ob
fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.plot(lam_ob * 1e3, spread_ob, color=SERIES[0], label=f"|D(λ)| · 2 km · Δλ,  Δλ ↔ ±0.75×baud = {F_BAND_GHZ:.1f} GHz (as in experiment 05)")
ax.plot(lam_ob * 1e3, spread_ob_crude, color=SERIES[2], ls="-.", lw=1.2, label="same with the crude one-sided Δλ ↔ 53.1 GHz")
ax.axhline(res["bit_period_ps"], color=SERIES[4], ls="--", lw=1.4, label=f"one bit period 1/53.125 GBd = {res['bit_period_ps']:.1f} ps")
ax.axhline(res["bit_period_ps"] / 4, color=SERIES[3], ls=":", lw=1.2, label="quarter bit period (eye still comfortably open)")
for lam_nm, c_ in ((1310, SERIES[1]), (1550, SERIES[4])):
    ax.plot([lam_nm], [res[f"spread_2km_{lam_nm}_ps"]], "o", color=c_, ms=7)
    ax.annotate(f"{lam_nm} nm: {res[f'spread_2km_{lam_nm}_ps']:.1f} ps", xy=(lam_nm, res[f"spread_2km_{lam_nm}_ps"]),
                xytext=(lam_nm - 60, res[f"spread_2km_{lam_nm}_ps"] + 5), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=c_))
ax.axvspan(1260, 1360, color=SERIES[1], alpha=0.10, label="O-band 1260–1360 nm")
ax.axvspan(1530, 1565, color=SERIES[4], alpha=0.10, label="C-band 1530–1565 nm")
ax.set_xlabel("vacuum wavelength λ₀ (nm)"); ax.set_ylabel("chromatic delay spread after 2 km  (ps)")
ax.set_ylim(0, 45)
ax.set_title(f"Why the O-band: 53 Gbaud over 2 km of silica spreads {res['spread_2km_1310_ps']:.0f} ps at 1310 nm "
             f"but {res['spread_2km_1550_ps']:.0f} ps (1.5 bit periods) at 1550 nm", fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "oband_spread_2km.png"); plt.close(fig)

# --- Fig D: term-by-term curvature (notes §10) ------------------------------
fig, axs = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)
ax = axs[0]
for (label, S_, S1_, S2_), c_ in zip(terms, SERIES):
    ax.plot(lam_sio2 * 1e3, S_, color=c_, label=label)
ax.plot(lam_sio2 * 1e3, sum(t[1] for t in terms), color=PALETTE["ink"], lw=1.2, ls="--", label="sum = n² − 1")
ax.set_ylabel("S_i(λ) = B_i λ²/(λ² − C_i)")
ax.set_ylim(-0.1, 1.75)
ax.set_title("Each Sellmeier term of silica: the two UV terms are ~flat, the IR term slopes down", fontsize=10)
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax = axs[1]
for (label, S_, S1_, S2_), c_ in zip(terms, SERIES):
    ax.plot(lam_sio2 * 1e3, S2_, color=c_, label=f"S''  {label}")
ax.axhline(0, color=PALETTE["ink2"], lw=0.8)
ax.set_ylabel("d²S_i/dλ²  (1/µm²)")
ax.set_title("Second derivatives: UV terms bend UP (+, falling as 1/λ⁴), the IR term bends DOWN (−, ~constant)", fontsize=10)
ax.set_ylim(-0.08, 0.25); ax.legend(fontsize=8, loc="upper right")
ax = axs[2]
ax.plot(lam_sio2 * 1e3, curv_uv, color=SERIES[0], label="UV part  S_uv''/(2n)")
ax.plot(lam_sio2 * 1e3, curv_ir, color=SERIES[2], label="IR part  S_ir''/(2n)")
ax.plot(lam_sio2 * 1e3, curv_sq, color=SERIES[3], label="−S'²/(4n³) (small)")
ax.plot(lam_sio2 * 1e3, curv_tot, color=PALETTE["ink"], lw=2.2, label="total d²n/dλ²")
ax.axhline(0, color=PALETTE["ink2"], lw=0.8)
ax.axvline(zdw * 1e3, color=SERIES[4], ls="-.", lw=1.2, label=f"cancellation → ZDW = {zdw*1e3:.0f} nm")
ax.set_ylim(-0.03, 0.06)
ax.set_xlabel("vacuum wavelength λ₀ (nm)"); ax.set_ylabel("d²n/dλ²  (1/µm²)")
ax.set_title("d²n/dλ² = S''/(2n) − S'²/(4n³): the ZDW is where UV (+) and IR (−) curvature cancel", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "term_curvature.png"); plt.close(fig)

# --- Fig E: small-change rule ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axs[0]
dl = np.linspace(-15, 15, 301)
f0 = C0 / (REF.lambda_nm * 1e-9)
exact = (C0 / ((REF.lambda_nm + dl) * 1e-9) - f0) / 1e9
lin = -scr * dl
ax.plot(dl, exact, color=SERIES[0], label="exact  Δf = c/(λ+Δλ) − c/λ")
ax.plot(dl, lin, color=SERIES[1], ls="--", label=f"linear  Δf = −(c/λ²)Δλ = −{scr:.1f} GHz/nm · Δλ")
ax.plot([REF.fsr_nm], [-scr * REF.fsr_nm], "o", color=SERIES[3], ms=6)
ax.annotate(f"one FSR: 10.3 nm ↔ {scr*REF.fsr_nm/1e3:.2f} THz", xy=(REF.fsr_nm, -scr * REF.fsr_nm), xytext=(-1, -2200), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=SERIES[3]))
cc = res["capstone_conversions"]
ax.text(0.03, 0.05, "capstone conversions at 1310 nm\n"
        f"  1 K = 50 pm  ↔  {cc['thermal_50pm_per_K_in_ghz_per_K']:.2f} GHz\n"
        f"  FWHM 374 pm  ↔  {cc['fwhm_374pm_in_ghz']:.1f} GHz\n"
        f"  δ_opt 108 pm  ↔  {cc['delta_opt_108pm_in_ghz']:.1f} GHz\n"
        f"  200 GHz channel  ↔  {cc['channel_spacing_200ghz_in_nm']:.3f} nm\n"
        f"  1 pm  ↔  {cc['one_pm_in_mhz']:.0f} MHz",
        transform=ax.transAxes, fontsize=8, family="monospace", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec=PALETTE["line"]))
ax.set_xlabel("Δλ from 1310 nm  (nm)"); ax.set_ylabel("Δf  (GHz)")
ax.set_title("At 1310 nm: 1 nm ↔ 174.7 GHz; the linear rule is exact to 1 % out to ±13 nm", fontsize=9.5)
ax.legend(fontsize=8)
ax = axs[1]
ax.plot(dl, (exact - lin), color=SERIES[0])
ax.set_xlabel("Δλ from 1310 nm  (nm)"); ax.set_ylabel("exact − linear  (GHz)")
ax.set_title("Rule error = 2nd-order term (c/λ³)Δλ²: 0.13 GHz at 1 nm, 30 GHz at 15 nm", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "small_change_rule.png"); plt.close(fig)

# --- Fig F: capstone FSR by index -------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
labels = list(fsr_from.keys()); vals = [fsr_from[k]["fsr_thz"] for k in labels]; nms = [fsr_from[k]["fsr_nm"] for k in labels]
bars = ax.bar(range(len(labels)), vals, color=[SERIES[4], SERIES[3], SERIES[1], SERIES[0]])
# Value labels sit with their bottom edge 0.08 THz above each bar (va="bottom"), so the label of the 1.80 THz bar
# starts at 1.88 THz, clear of the dashed 1.8 THz line; the index itself goes into the tick label.
for i, (v, nmv, k) in enumerate(zip(vals, nms, labels)):
    ax.text(i, v + 0.08, f"{v:.2f} THz\n{nmv:.1f} nm", ha="center", va="bottom", fontsize=8.5)
ax.axhline(REF.fsr_thz, color=PALETTE["ink"], ls="--", lw=1.2, label=f"measured FSR {REF.fsr_thz} THz (Lightmatter OFC 2025)")
ax.set_xticks(range(len(labels))); ax.set_xticklabels([f"{k}\nindex = {fsr_from[k]['index']:.2f}" for k in labels], fontsize=8.5)
ax.set_ylabel("FSR = c / (index · L)  (THz),  L = 39.6 µm"); ax.set_ylim(0, 3.6)
ax.set_title("Only the waveguide group index reproduces the measured ring FSR; n_eff overestimates it by 1.7×", fontsize=9.5)
ax.legend(fontsize=8); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT / "capstone_fsr.png"); plt.close(fig)
say("figures written")

# ==========================================================================
# 4. Video: the tangent sliding along n(λ) traces n_g and D
# ==========================================================================
say("rendering video ...")
lam_anim = np.linspace(0.65, 1.95, 240)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
ax0, ax1, ax2 = axs
lw = np.linspace(0.6, 2.0, 600)
ax0.plot(lw, silica.n(lw), color=SERIES[0], label="n(λ)")
tan_line, = ax0.plot([], [], color=SERIES[1], ls="--", lw=1.4, label="tangent")
pt, = ax0.plot([], [], "o", color=SERIES[0], ms=6)
icpt, = ax0.plot([], [], "s", color=SERIES[1], ms=7, label="intercept = n_g")
ax0.set_xlim(-0.05, 2.05); ax0.set_ylim(1.43, 1.49)
ax0.set_xlabel("λ₀ (µm)"); ax0.set_ylabel("index"); ax0.legend(fontsize=8, loc="upper right")
ax0.set_title("Silica n(λ): slide the tangent, read n_g at λ = 0", fontsize=9.5)
ax1.plot(lw, silica.ng(lw), color=PALETTE["line"], lw=1)
ng_trace, = ax1.plot([], [], color=SERIES[2], lw=2)
ng_pt, = ax1.plot([], [], "s", color=SERIES[1], ms=7)
ax1.set_xlim(0.6, 2.0); ax1.set_ylim(1.455, 1.49)
ax1.set_xlabel("λ₀ (µm)"); ax1.set_ylabel("n_g"); ax1.set_title("n_g = n − λ dn/dλ: minimum where D = 0", fontsize=9.5)
ax2.plot(lw, silica.D_ps_nm_km(lw), color=PALETTE["line"], lw=1); ax2.axhline(0, color=PALETTE["ink2"], lw=0.8)
D_trace, = ax2.plot([], [], color=SERIES[3], lw=2)
D_pt, = ax2.plot([], [], "o", color=SERIES[3], ms=7)
ax2.set_xlim(0.6, 2.0); ax2.set_ylim(-250, 60)
ax2.set_xlabel("λ₀ (µm)"); ax2.set_ylabel("D (ps/(nm·km))"); ax2.set_title("D = (1/c) dn_g/dλ: slope of the n_g curve", fontsize=9.5)
txt = ax0.text(0.02, 0.05, "", transform=ax0.transAxes, fontsize=8.5, family="monospace")
fig.tight_layout()


def update(i):
    lt = lam_anim[i]
    nt, st = float(silica.n(lt)), float(silica.dn_dlam(lt))
    xs = np.array([0.0, 2.0]); tan_line.set_data(xs, nt + st * (xs - lt))
    pt.set_data([lt], [nt]); icpt.set_data([0], [nt - st * lt])
    sub = lam_anim[: i + 1]
    ng_trace.set_data(sub, silica.ng(sub)); ng_pt.set_data([lt], [nt - st * lt])
    D_trace.set_data(sub, silica.D_ps_nm_km(sub)); D_pt.set_data([lt], [float(silica.D_ps_nm_km(lt))])
    txt.set_text(f"λ = {lt*1e3:6.0f} nm\nn   = {nt:.4f}\nn_g = {nt - st*lt:.4f}\nD   = {float(silica.D_ps_nm_km(lt)):+7.1f} ps/(nm·km)")
    return tan_line, pt, icpt, ng_trace, ng_pt, D_trace, D_pt, txt


anim = FuncAnimation(fig, update, frames=len(lam_anim), blit=False)
anim.save(OUT / "tangent_sliding.mp4", writer=FFMpegWriter(fps=30, bitrate=2400))
plt.close(fig)
# contact sheet
fig, axs = plt.subplots(2, 3, figsize=(13, 7.0))
for ax, lt in zip(axs.ravel(), [0.65, 0.9, 1.15, zdw, 1.55, 1.95]):
    nt, st = float(silica.n(lt)), float(silica.dn_dlam(lt))
    h_n, = ax.plot(lw, silica.n(lw), color=SERIES[0], label="n(λ), Malitson silica")
    xs = np.array([0.0, 2.0])
    h_t, = ax.plot(xs, nt + st * (xs - lt), color=SERIES[1], ls="--", lw=1.3, label="tangent to n(λ) at the marked λ")
    h_p, = ax.plot([lt], [nt], "o", color=SERIES[0], ms=5, label="tangent point (λ, n(λ))")
    h_i, = ax.plot([0], [nt - st * lt], "s", color=SERIES[1], ms=6, label="intercept at λ = 0: n − λ dn/dλ = n_g")
    ax.set_xlim(-0.05, 2.05); ax.set_ylim(1.43, 1.49)
    ax.set_title(f"λ = {lt*1e3:.0f} nm: n_g = {nt - st*lt:.4f}, D = {float(silica.D_ps_nm_km(lt)):+.0f} ps/(nm·km)", fontsize=8.5)
    ax.set_xlabel("λ₀ (µm)"); ax.set_ylabel("index")
fig.suptitle("Frames of tangent_sliding.mp4: the intercept (square) is n_g; it is lowest at the zero-dispersion wavelength", fontsize=10)
fig.legend(handles=[h_n, h_t, h_p, h_i], loc="lower center", ncol=4, fontsize=8.5, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "tangent_sliding_frames.png"); plt.close(fig)
say("video written")

# ==========================================================================
# 5. Data tables and JSON
# ==========================================================================
np.savetxt(OUT / "silica_dispersion_table.csv",
           np.column_stack([lam_sio2 * 1e3, silica.n(lam_sio2), silica.ng(lam_sio2), Dsi, silica.beta2_ps2_km(lam_sio2)]),
           delimiter=",", header="lambda_nm,n,n_g,D_ps_nm_km,beta2_ps2_km", comments="", fmt="%.6f")
np.savetxt(OUT / "silicon_dispersion_table.csv",
           np.column_stack([lam_si * 1e3, silicon.n(lam_si), silicon.ng(lam_si), silicon.D_ps_nm_km(lam_si), silicon.beta2_ps2_km(lam_si)]),
           delimiter=",", header="lambda_nm,n,n_g,D_ps_nm_km,beta2_ps2_km", comments="", fmt="%.6f")

expect = {
    "n_silica_1310": {"expected": 1.4468, "source": "Malitson value quoted in the brief"},
    "ng_silica_1310": {"expected": 1.4620, "source": "bulk silica group index tables (refractiveindex.info, Malitson)"},
    "zdw_silica_um": {"expected": 1.27, "source": "brief / notes §10-11 (≈1.27 µm bulk silica)"},
    "D_silica_1310_ps_nm_km": {"expected": round(res["D_silica_1310_omega_route_ps_nm_km"], 4),
                               "source": "independent ω-route: β₂ = d²β/dω² by finite differences in ω, D = −2πc β₂/λ² (no λ-derivatives of n); "
                                         "bulk silica ≈ +3.5 ps/(nm·km), which G.652 fibre's waveguide dispersion cancels to put its ZDW near 1310 nm. "
                                         "Linear extrapolation from the ZDW (3.72) is a cross-check only: its 4 % excess is the curvature of D(λ)."},
    "D_silica_1550_ps_nm_km": {"expected": 21.0, "source": "brief: bulk silica 20-22 ps/(nm·km); fibre 17 includes waveguide dispersion"},
    "small_change_ghz_per_nm_1310": {"expected": 174.7, "source": "brief / small-change rule"},
    "fsr_thz_ng42": {"expected": REF.fsr_thz, "source": "REF.fsr_thz (Lightmatter)"},
    "fsr_nm_ng42": {"expected": REF.fsr_nm, "source": "REF.fsr_nm"},
    "n_silicon_1310": {"expected": REF.n_si, "source": "REF.n_si (rounded 3.50)"},
    "ng_silicon_1310": {"expected": 3.70, "source": "bulk crystalline silicon group index ≈ 3.7 at 1.31 µm (Li 1980 / refractiveindex.info)"},
    "dlambda_dT_pm_per_K_from_ng": {"expected": REF.dlambda_dT_pm_per_K, "source": "REF.dlambda_dT_pm_per_K"},
}
headline = {}
for key, e in expect.items():
    v = res[key]; agree = 100.0 * (1 - abs(v - e["expected"]) / abs(e["expected"]))
    headline[key] = {"value": v, "expected": e["expected"], "agreement_percent": round(agree, 2), "source": e["source"]}
res["headline"] = headline
res["runtime_seconds"] = time.time() - T0
try:
    ffmpeg_ver = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-version"], capture_output=True, text=True, timeout=20).stdout.split("\n")[0].split(" ")[2]
except Exception:
    ffmpeg_ver = "unknown"
try:
    import ipywidgets, nbformat
    ipyw_ver, nbf_ver = ipywidgets.__version__, nbformat.__version__
except Exception:
    ipyw_ver = nbf_ver = "unknown"
res["versions"] = {"python": platform.python_version(), "sympy": sp.__version__, "numpy": np.__version__,
                   "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "ffmpeg": ffmpeg_ver,
                   "ipywidgets": ipyw_ver, "nbformat": nbf_ver}
(OUT / "results.json").write_text(json.dumps(res, indent=2))

# ---- tools.json: same content as the README's "Tools used" subsections ------
hd = headline
tools = [
    {"tool": "sympy", "version": sp.__version__,
     "what_it_is": "A Python computer-algebra system: symbolic differentiation, simplification, series expansion and pretty-printing of equations. Normally used to derive and check formulas before they are coded numerically.",
     "used_for": "Deriving n_g = c dk/dω = n − λ dn/dλ and D = d(n_g/c)/dλ = −(λ/c) d²n/dλ² from k(ω) = n(λ(ω)) ω/c, the chain rule dn/dλ = S'/(2n), d²n/dλ² = S''/(2n) − S'²/(4n³), the UV/IR series of a single Sellmeier term (notes §10), and then lambdifying the exact Sellmeier n(λ), dn/dλ, d²n/dλ² so every number in this experiment is the evaluated symbolic derivation.",
     "result": f"out/derivation.txt: both identities are checked to simplify to 0. Series: d²S_u/dλ² ≈ 6B_uλ_u²/λ⁴ > 0 (UV bends up), d²S_ir/dλ² ≈ −2B_ir/λ_ir² < 0 (IR bends down). Lambdified silica n(1310 nm) = {hd['n_silica_1310']['value']:.5f} vs Malitson 1.4468 ({hd['n_silica_1310']['agreement_percent']:.2f} %).",
     "how_to_observe": "cd experiments/04_sellmeier_dispersion && ../../.venv/bin/python run.py ; read out/derivation.txt (section 1 of the log) and out/equations.png. To try another material, add a Sellmeier n_expr in sellmeier.py and wrap it in Material(); the derivatives are generated automatically."},
    {"tool": "numpy + scipy", "version": f"numpy {np.__version__}, scipy {scipy.__version__}",
     "what_it_is": "numpy is the array library every scientific Python code is built on; scipy adds numerical algorithms (root finding, optimisation, integration). Used here for evaluating the lambdified expressions on wavelength grids and for the root finder scipy.optimize.brentq.",
     "used_for": "Evaluating n, n_g, D on 0.6–2.0 µm (silica) and 1.2–2.0 µm (silicon), finite-difference cross-checks of the sympy derivatives, brentq root of d²n/dλ² for the zero-dispersion wavelength, the term-by-term decomposition of d²n/dλ² at the ZDW, the small-change rule and the ring FSR / thermal-shift numbers.",
     "result": f"ZDW(silica) = {hd['zdw_silica_um']['value']*1e3:.1f} nm (expected ≈1270, {hd['zdw_silica_um']['agreement_percent']:.2f} %); D(1550) = {hd['D_silica_1550_ps_nm_km']['value']:.2f} ps/(nm·km) (bulk 20–22); D(1310) = {hd['D_silica_1310_ps_nm_km']['value']:.3f} ps/(nm·km) vs the independent ω-route {hd['D_silica_1310_ps_nm_km']['expected']:.3f} ({hd['D_silica_1310_ps_nm_km']['agreement_percent']:.2f} %); n_g(1310, silica) = {hd['ng_silica_1310']['value']:.4f}; small-change rule {hd['small_change_ghz_per_nm_1310']['value']:.2f} GHz/nm (174.7); FSR = {hd['fsr_thz_ng42']['value']:.3f} THz / {hd['fsr_nm_ng42']['value']:.2f} nm with n_g = 4.2 (1.8 / 10.3); 2 km delay spread across ±0.75×baud {res['spread_2km_1310_ps']:.1f} ps at 1310 nm vs {res['spread_2km_1550_ps']:.1f} ps at 1550 nm (experiment 05: 3.3 / 28.0). Finite-difference vs symbolic: max|Δn_g| = {res['fd_check_ng_max_abs_diff']:.1e}.",
     "how_to_observe": "out/results.json (all numbers), out/results.txt (the printed log), out/silica_dispersion_table.csv and out/silicon_dispersion_table.csv (λ, n, n_g, D, β₂ tables). Change L0 / REF.ng in run.py or the bracket of brentq to explore."},
    {"tool": "matplotlib (+ ffmpeg)", "version": f"matplotlib {matplotlib.__version__}, ffmpeg {ffmpeg_ver}",
     "what_it_is": "The standard Python plotting library; FuncAnimation + FFMpegWriter drive the ffmpeg encoder to turn a sequence of frames into an mp4. Also used here to render the equations with mathtext because there is no LaTeX on this machine.",
     "used_for": "Eleven PNG files: ten figures (equations, n and n_g, two tangent-intercept constructions, D vs λ, term-by-term curvature, the Sellmeier poles on a log axis, the small-change rule, the ring-FSR-by-index bar chart, the 2 km spread vs λ) plus the contact sheet of the 8 s video tangent_sliding.mp4.",
     "result": "out/*.png (11 files), out/tangent_sliding.mp4 (240 frames at 30 fps, 8.0 s), out/tangent_sliding_frames.png. In the video the square (the λ = 0 intercept of the tangent) is n_g and is lowest exactly where D crosses zero.",
     "how_to_observe": "open out/tangent_sliding.mp4; the figures are embedded in README.md. Edit lam_anim in run.py to change the sweep range or fps in FFMpegWriter(fps=30)."},
    {"tool": "Jupyter notebook + ipywidgets", "version": f"nbformat {nbf_ver}, ipywidgets {ipyw_ver}, kernel photonics-sims",
     "what_it_is": "Jupyter notebooks mix code, output and text; ipywidgets adds sliders that re-run a Python function when moved. Used for interactive exploration where a script would need a re-run per parameter.",
     "used_for": "explore.ipynb (built by make_notebook.py, executed and saved with outputs): sliders that (1) slide the tangent along n(λ) for silica or silicon and read n_g from the intercept, (2) move the IR Sellmeier pole (λ_ir, B_ir) and scale the UV strengths and watch the zero-dispersion wavelength move, (3) compute the ring FSR from n_g and L and draw the eight 200 GHz channels inside one FSR, (4) convert pm ↔ GHz with the small-change rule. A fifth cell asserts the notebook's evaluation of the same Material objects equals out/results.json to 1e-12.",
     "result": "Executed outputs saved in explore.ipynb (static figures for the default slider values). Moving λ_ir from 9.9 µm to 6 µm pulls the ZDW from 1273 nm to 978 nm (to 14 µm: 1523 nm); doubling B_ir to 1.4 pulls it to 1145 nm; scaling the UV strengths by 1.1 pushes it to 1302 nm. Increasing n_g from 3.7 (bulk Si) to 4.2 (strip) moves the FSR from 2.05 THz to 1.80 THz; the eight 200 GHz channels span 8.01 nm and fit inside the 10.32 nm FSR.",
     "how_to_observe": "cd experiments/04_sellmeier_dispersion && ../../.venv/bin/jupyter lab explore.ipynb and drag the sliders. run.py rebuilds and re-executes the notebook headless at the end of every run (make_notebook.py + jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=photonics-sims), so it never goes stale. The step is load-sensitive (kernel start + 11 cells: ≈ 35 s on an idle laptop, many minutes when other solver runs saturate the CPU); the per-cell timeout is 60 s, a stalled attempt is retried once, and a third attempt skips the four slider cells (tag widgets); set SKIP_NOTEBOOK=1 to skip it and keep the headless part alone at ≈ 1 min. By hand: ../../.venv/bin/python make_notebook.py && ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb."},
]
(OUT / "tools.json").write_text(json.dumps(tools, indent=2, ensure_ascii=False))

say("\nHEADLINE NUMBERS")
for k_, hv in headline.items():
    say(f"  {k_:32s} = {hv['value']:12.5f}   expected {hv['expected']:10.4f}   agreement {hv['agreement_percent']:7.2f} %")
say(f"\nfigures, video, tables and JSON finished in {res['runtime_seconds']:.1f} s")
# The log is written here, before the notebook step, so out/results.txt exists even if nbconvert is slow or fails;
# it is rewritten with the notebook timing at the very end.
(OUT / "results.txt").write_text("\n".join(LOG) + "\n")

# ==========================================================================
# 6. Rebuild and re-execute explore.ipynb so its saved outputs (and its 1e-12
#    assertion against out/results.json) can never go stale after a parameter change.
#    This step is load-sensitive (kernel start + 11 cells; ≈ 35 s idle, many minutes when other
#    solver runs saturate the CPU).  SKIP_NOTEBOOK=1 skips it; a failure is recorded, not fatal.
# ==========================================================================
HERE = pathlib.Path(__file__).resolve().parent
PY = pathlib.Path(sys.executable)                  # the .venv interpreter this script runs under
JUPYTER = PY.parent / "jupyter"
if os.environ.get("SKIP_NOTEBOOK", "") not in ("", "0"):
    res["notebook_status"] = "skipped (SKIP_NOTEBOOK set); explore.ipynb left as it was"
    say("\nSKIP_NOTEBOOK set: explore.ipynb NOT rebuilt or re-executed (its saved outputs may lag results.json)")
else:
    say("\nrebuilding and executing explore.ipynb ...")
    t_nb = time.time()
    try:
        subprocess.run([str(PY), "make_notebook.py"], cwd=HERE, check=True)
        # Every cell of explore.ipynb runs in a few seconds, so the 60 s per-cell timeout only ever fires when the
        # kernel↔nbconvert messaging stalls.  On this machine (ipykernel 7.3.0, ipywidgets 8.1.9, jupyter_client 8.10)
        # that stall is intermittent (~1 attempt in 2: the kernel goes idle and never answers the next execute
        # request, always shortly after an ipywidgets `interact` cell) and was never seen with the four slider cells
        # removed (0 of 4).  So: two attempts with every cell, then one attempt that skips the cells tagged "widgets"
        # (the static figure cells and the results.json cross-check still execute).  What happened is recorded in
        # results.json → notebook_status.
        nb_base = [str(JUPYTER), "nbconvert", "--to", "notebook", "--execute", "--inplace",
                   "--ExecutePreprocessor.timeout=60", "--ExecutePreprocessor.kernel_name=photonics-sims"]
        attempts = [("all cells", []), ("all cells, fresh kernel", []),
                    ("cells tagged 'widgets' skipped", ["--ExecutePreprocessor.skip_cells_with_tag=widgets"])]
        for n_try, (label, extra) in enumerate(attempts, start=1):
            t_try = time.time()
            proc = subprocess.run(nb_base + extra + ["explore.ipynb"], cwd=HERE)
            if proc.returncode == 0:
                break
            say(f"nbconvert attempt {n_try} ({label}) failed (returncode {proc.returncode}) after {time.time() - t_try:.0f} s")
        else:
            raise RuntimeError("jupyter nbconvert failed three times (see the tracebacks above)")
        res["notebook_status"] = f"executed on attempt {n_try} of 3 ({label})"
        res["notebook_widget_cells_executed"] = (label != attempts[-1][0])
        say(f"explore.ipynb rebuilt and executed in {time.time() - t_nb:.1f} s (attempt {n_try}: {label})")
    except Exception as exc:                        # keep the headless outputs valid; report the failure
        res["notebook_status"] = f"FAILED: {exc!r}"
        say(f"explore.ipynb step FAILED after {time.time() - t_nb:.1f} s: {exc!r}")
    res["notebook_seconds"] = time.time() - t_nb
res["runtime_seconds_total_with_notebook"] = time.time() - T0
(OUT / "results.json").write_text(json.dumps(res, indent=2))   # add the timing keys; headline numbers unchanged
say(f"\nrun.py finished in {res['runtime_seconds_total_with_notebook']:.1f} s total")
(OUT / "results.txt").write_text("\n".join(LOG) + "\n")
if not res["notebook_status"].startswith(("executed", "skipped")):
    sys.exit(1)
