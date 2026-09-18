"""SAX part of experiment 11 (run with the .venv interpreter; run.py calls it via subprocess).

What it does (notes §26, §27 and the capstone):
  1. The 2x2 directional-coupler scattering matrix  S = [[t, -jK], [-jK, t]],  t = cos(κ_c L),
     K = sin(κ_c L), built as a SAX model; unitarity S†S = I and reciprocity checked numerically.
  2. |t|², |K|² versus coupler length L for the three FDTD gaps (κ_c from the analytic supermodes
     of slab_coupler.py) -> the coupling length L_c = π/(2κ_c) and the 3 dB length L_c/2.
  3. Interference of two coherent inputs s1, s2 = e^{jφ}: |s1 + s2|² versus |s1|² + |s2|² as a
     function of the relative phase φ (notes §26: the cross term 2 Re{s1 s2*}), once directly with
     the S-matrix and once as a SAX circuit (two 50/50 couplers + a phase shifter = Mach-Zehnder).
  4. Capstone: the ring-bus point coupler.  κ² = sin²(κ_c L_eff) with L_eff = sqrt(2πR/γ); the
     reference κ² = 0.107 fixes κ_c; a gap error Δg multiplies κ_c by e^{-γ Δg}; a SAX ring circuit
     (coupler + 39.6 µm doped waveguide, a = 0.945) then shows how the on-resonance transmission
     T_min = ((t-a)/(1-ta))² leaves zero when t ≠ a: "critical coupling is a fabrication lottery".

Writes out/sax_results.json and the figures listed in FIGS.
"""
import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
from common import REF, use_style, SERIES, PALETTE
from common.params import k0_per_um
import slab_coupler as sc

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
use_style()
LAM = REF.lambda_nm * 1e-3                     # 1.31 µm
K0 = k0_per_um(REF.lambda_nm)                  # rad/µm
# ONE round-trip amplitude retention a for the whole file: the closed-form T_min = ((t-a)/(1-ta))² and the
# SAX waveguide model (through its loss in dB/cm) both derive from A_RT, so changing it here changes both.
# REF.a_round_trip = 0.945 = REF.t_coupler (critical coupling); REF.loss_db_cm_doped = 125 dB/cm would give
# a = 10^(-125·39.6e-4/20) = 0.9446 instead, which is recorded in sax_results.json["round_trip"] for reference.
A_RT = REF.a_round_trip                                                   # set 0.9986 for the passive 3 dB/cm ring
LOSS_DB_CM = -20 * np.log10(A_RT) / (REF.round_trip_um * 1e-4)           # 124.1 dB/cm reproduces a = A_RT exactly
FIGS = ["sax_coupler_vs_length.png", "sax_interference.png", "critical_coupling_lottery.png"]


# ----------------------------------------------------------------------------- SAX models
def coupler_kL(kL=0.5):
    """Directional coupler with accumulated coupling angle kL = κ_c·L  (notes §27)."""
    t = jnp.cos(kL); k = jnp.sin(kL)
    return sax.reciprocal({("in0", "out0"): t, ("in0", "out1"): -1j * k,
                           ("in1", "out0"): -1j * k, ("in1", "out1"): t})


def coupler(coupling=REF.kappa2):
    """Same coupler parametrised by power coupling κ² (the brief's snippet)."""
    k = coupling ** 0.5; t = (1 - coupling) ** 0.5
    return sax.reciprocal({("in0", "out0"): t, ("in0", "out1"): -1j * k,
                           ("in1", "out0"): -1j * k, ("in1", "out1"): t})


def phase(phi=0.0):
    """Lossless phase shifter e^{-jφ} (a +z wave picks up e^{-jβz}, notes §3)."""
    return sax.reciprocal({("in0", "out0"): jnp.exp(-1j * phi)})


def waveguide(wl=1.31, length=REF.round_trip_um, neff=REF.neff, ng=REF.ng, wl0=1.31,
              loss_db_cm=LOSS_DB_CM):
    n = neff - (wl - wl0) * (ng - neff) / wl0
    amp = 10 ** (-loss_db_cm * length * 1e-4 / 20)
    return sax.reciprocal({("in0", "out0"): amp * jnp.exp(-2j * jnp.pi * n * length / wl)})


def smatrix(sdict, ins=("in0", "in1"), outs=("out0", "out1")):
    """2x2 numpy matrix S[out, in] from a SAX sdict (s_out = S s_in, notes §26)."""
    return np.array([[complex(sdict[(i, o)]) for i in ins] for o in outs])


# ----------------------------------------------------------------------------- 1. unitarity
res = {"round_trip": dict(a_used=float(A_RT), loss_db_cm_used=float(LOSS_DB_CM),
                          a_from_REF_loss_db_cm_doped=float(10 ** (-REF.loss_db_cm_doped * REF.round_trip_um * 1e-4 / 20)),
                          REF_loss_db_cm_doped=REF.loss_db_cm_doped,
                          note="one a feeds both the closed-form T_min and the SAX waveguide loss")}
kL_test = np.linspace(0, np.pi, 25)
uni_err = max(np.abs(smatrix(coupler_kL(kL)).conj().T @ smatrix(coupler_kL(kL)) - np.eye(2)).max()
              for kL in kL_test)
S = smatrix(coupler_kL(np.pi / 4))
res["unitarity"] = dict(max_abs_deviation_SdagS_from_I=float(uni_err),
                        reciprocity_max_abs_S_minus_ST=float(np.abs(S - S.T).max()),
                        S_50_50=[[str(np.round(v, 4)) for v in row] for row in S],
                        cross_phase_deg=float(np.degrees(np.angle(S[1, 0]))))
print(f"[sax] unitarity max |S†S - I| = {uni_err:.2e}; 50/50 cross phase = {res['unitarity']['cross_phase_deg']:.1f} deg")

# ----------------------------------------------------------------------------- 2. |t|², |K|² vs L
gaps = [0.15, 0.20, 0.30]
L = np.linspace(0, 40, 801)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
res["vs_length"] = {}
for i, g in enumerate(gaps):
    kc = sc.supermodes(g, lam=LAM)["kappa"]
    sd = coupler_kL(kL=jnp.asarray(kc * L))
    T = np.abs(np.asarray(sd[("in0", "out0")])) ** 2
    K2 = np.abs(np.asarray(sd[("in0", "out1")])) ** 2
    Lc = np.pi / (2 * kc)
    ax.plot(L, T, color=SERIES[i], label=f"gap {g*1e3:.0f} nm: |t|² (stays in guide 1), L_c = {Lc:.1f} µm")
    ax.plot(L, K2, color=SERIES[i], ls="--", label=f"gap {g*1e3:.0f} nm: |K|² (crossed to guide 2)")
    ax.axvline(Lc, color=SERIES[i], lw=0.8, alpha=0.5)
    res["vs_length"][f"gap_{g*1e3:.0f}nm"] = dict(kappa_c_per_um=float(kc), L_c_um=float(Lc),
                                                  L_3dB_um=float(Lc / 2), sum_check=float(np.abs(T + K2 - 1).max()))
ax.axhline(0.5, color=PALETTE["muted"], lw=0.8, ls=":")
ax.set_xlabel("coupler length L (µm)"); ax.set_ylabel("output power fraction")
ax.set_title("A 2×2 coupler splits power as cos²(κ_c L) / sin²(κ_c L); the 50/50 point is at L_c/2")
ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18)); ax.set_xlim(0, 40); ax.set_ylim(-0.02, 1.02)
fig.tight_layout(); fig.savefig(OUT / FIGS[0], bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- 3. interference
phi = np.linspace(-np.pi, np.pi, 361)
S50 = smatrix(coupler_kL(np.pi / 4))
s_in = np.stack([np.ones_like(phi), np.exp(1j * phi)]) / np.sqrt(2)     # |s1|² + |s2|² = 1 W
s_out = S50 @ s_in
P3, P4 = np.abs(s_out[0]) ** 2, np.abs(s_out[1]) ** 2
naive_sum = np.abs(s_in[0]) ** 2 + np.abs(s_in[1]) ** 2                   # = 1 always
coherent = np.abs(s_in[0] + s_in[1]) ** 2                                  # 1 + cos φ
cross = 2 * np.real(s_in[0] * np.conj(s_in[1]))
# same thing as a SAX circuit: 50/50 coupler -> phase shifter in one arm -> 50/50 coupler (MZI)
mzi, _ = sax.circuit(
    netlist={"instances": {"c1": "coupler", "ps": "phase", "wg": "phase", "c2": "coupler"},
             "connections": {"c1,out0": "ps,in0", "c1,out1": "wg,in0", "ps,out0": "c2,in0", "wg,out0": "c2,in1"},
             "ports": {"in0": "c1,in0", "in1": "c1,in1", "out0": "c2,out0", "out1": "c2,out1"}},
    models={"coupler": coupler_kL, "phase": phase})
Smzi = mzi(c1={"kL": np.pi / 4}, c2={"kL": np.pi / 4}, ps={"phi": jnp.asarray(phi)}, wg={"phi": 0.0})
mzi_bar = np.abs(np.asarray(Smzi[("in0", "out0")])) ** 2
mzi_cross = np.abs(np.asarray(Smzi[("in0", "out1")])) ** 2
# in an MZI the second coupler sees s1 = -j/√2·..., i.e. a fixed extra 90°: bar = sin²(φ/2), cross = cos²(φ/2)
mzi_err = float(max(np.abs(mzi_bar - np.sin(phi / 2) ** 2).max(), np.abs(mzi_cross - np.cos(phi / 2) ** 2).max()))
res["interference"] = dict(phi_deg_for_all_power_in_out0=float(np.degrees(phi[np.argmax(P3)])),
                           phi_deg_for_all_power_in_out1=float(np.degrees(phi[np.argmax(P4)])),
                           max_P3=float(P3.max()), min_P3=float(P3.min()),
                           total_out_minus_total_in_max=float(np.abs(P3 + P4 - 1).max()),
                           coherent_sum_range=[float(coherent.min()), float(coherent.max())],
                           mzi_vs_formula_max_err=mzi_err)
print(f"[sax] all power exits out0 at φ = {res['interference']['phi_deg_for_all_power_in_out0']:.0f} deg; "
      f"P3+P4-1 max = {res['interference']['total_out_minus_total_in_max']:.1e}; MZI vs sin²/cos² err = {mzi_err:.1e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4.1))
ax = axs[0]
ax.plot(np.degrees(phi), naive_sum, color=SERIES[0], label="|s₁|² + |s₂|²  (adding powers)")
ax.plot(np.degrees(phi), coherent, color=SERIES[1], label="|s₁ + s₂|²  (adding amplitudes)")
ax.plot(np.degrees(phi), cross, color=SERIES[2], ls="--", label="cross term 2 Re{s₁ s₂*}")
ax.set_xlabel("relative phase φ of the two inputs (deg)"); ax.set_ylabel("power (W, inputs ½ W each)")
ax.set_title("Amplitudes add, not powers: the cross term\nswings the total between 0 and 2× (notes §26)", fontsize=9.5)
ax.set_xlim(-180, 180)
ax = axs[1]
ax.plot(np.degrees(phi), P3, color=SERIES[0], label="P₃ = |t s₁ − jK s₂|²")
ax.plot(np.degrees(phi), P4, color=SERIES[1], label="P₄ = |−jK s₁ + t s₂|²")
ax.plot(np.degrees(phi), P3 + P4, color=PALETTE["muted"], ls=":", label="P₃ + P₄ (unitary: always 1 W)")
ax.set_xlabel("relative phase φ of the two inputs (deg)"); ax.set_ylabel("output power (W)")
ax.set_title("Two coherent ½ W inputs into one 50/50 coupler: the\nphase steers all 1 W to one port (φ = ±90°)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1); ax.set_xlim(-180, 180)   # below: P₃, P₄ cross at φ = 0
ax = axs[2]
ax.plot(np.degrees(phi), np.cos(phi / 2) ** 2, color=SERIES[0], label="cross port, formula cos²(φ/2)")
ax.plot(np.degrees(phi), np.sin(phi / 2) ** 2, color=SERIES[1], label="bar port, formula sin²(φ/2)")
ax.plot(np.degrees(phi), mzi_cross, "o", ms=3.5, markevery=18, color=SERIES[0], label="SAX circuit, cross port")
ax.plot(np.degrees(phi), mzi_bar, "s", ms=3.5, markevery=18, color=SERIES[1], label="SAX circuit, bar port")
ax.set_xlabel("phase φ added in one arm (deg)"); ax.set_ylabel("output power (W, 1 W in)")
ax.set_title("Same physics as a SAX circuit (Mach-Zehnder): coupler →\nphase in one arm → coupler; the first coupler adds its own −90°", fontsize=9.5)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); ax.set_xlim(-180, 180)
axs[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
fig.tight_layout(); fig.savefig(OUT / FIGS[1], bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- 4. capstone point coupler
R = REF.radius_um
g_slab = sc.single_slab_te(lam=LAM)["gamma"]                                # 2-D 220 nm slab, 12.53 /µm
g_text = K0 * np.sqrt(REF.neff ** 2 - REF.n_sio2 ** 2)                        # textbook n_eff 2.5 -> 9.77 /µm
cap = {}
for name, gam in (("textbook_neff2p5", g_text), ("slab2d_neff2p99", g_slab)):
    Leff = sc.point_coupler_length(R, gam)
    kL0 = np.arcsin(np.sqrt(REF.kappa2))          # coupling angle that gives κ² = 0.107
    kc0 = kL0 / Leff
    dg = np.array([-0.010, 0.0, +0.010])            # gap error in µm
    kappa2 = np.sin(kL0 * np.exp(-gam * dg)) ** 2
    t = np.sqrt(1 - kappa2)
    a = A_RT
    Tmin = ((t - a) / (1 - t * a)) ** 2
    # inverse problem: which gap error alone would give the reference T_min = 0.016?  ((t-a)/(1-ta))² = T_min has two
    # roots, t = (a ± s)/(1 ± s a) with s = √T_min (over-coupled t < a: smaller gap; under-coupled t > a: larger gap);
    # then κ² = 1 − t², κ_c L_eff = asin(√κ²) and Δg = −ln(asin(√κ²)/asin(√κ²_ref))/γ
    s_ref = np.sqrt(REF.t_min)
    t_roots = {"over_coupled_smaller_gap": (a - s_ref) / (1 - s_ref * a), "under_coupled_larger_gap": (a + s_ref) / (1 + s_ref * a)}
    dg_ref = {k: float(-1e3 * np.log(np.arcsin(np.sqrt(1 - tr ** 2)) / kL0) / gam) for k, tr in t_roots.items()}
    Tmin_check = max(abs(((tr - a) / (1 - tr * a)) ** 2 - REF.t_min) for tr in t_roots.values())
    cap[name] = dict(gamma_per_um=float(gam), decay_length_nm=float(1e3 / gam), L_eff_um=float(Leff), a_used=float(a),
                     gap_error_nm_for_ref_Tmin=dg_ref, t_for_ref_Tmin={k: float(v) for k, v in t_roots.items()},
                     ref_Tmin=REF.t_min, ref_Tmin_dB=float(-10 * np.log10(REF.t_min)), Tmin_roundtrip_check=float(Tmin_check),
                     kappa_c_for_ref_kappa2_per_um=float(kc0), kappa2_at_gap_err_nm={f"{int(d*1e3):+d}": float(k) for d, k in zip(dg, kappa2)},
                     kappa2_pct_change_per_10nm=[float((kappa2[0] / kappa2[1] - 1) * 100), float((kappa2[2] / kappa2[1] - 1) * 100)],
                     small_signal_pct_per_10nm=float(2 * gam * 0.010 * 100),
                     t_at_gap_err_nm={f"{int(d*1e3):+d}": float(v) for d, v in zip(dg, t)},
                     Tmin_at_gap_err_nm={f"{int(d*1e3):+d}": float(v) for d, v in zip(dg, Tmin)},
                     extinction_dB_at_gap_err_nm={f"{int(d*1e3):+d}": float(-10 * np.log10(max(v, 1e-12))) for d, v in zip(dg, Tmin)})
    print(f"[sax] {name}: γ = {gam:.2f}/µm, L_eff = {Leff:.2f} µm, κ_c = {kc0:.3f}/µm; κ² at gap -10/0/+10 nm = "
          f"{kappa2[0]:.3f}/{kappa2[1]:.3f}/{kappa2[2]:.3f} ({cap[name]['kappa2_pct_change_per_10nm'][0]:+.0f}%, "
          f"{cap[name]['kappa2_pct_change_per_10nm'][1]:+.0f}%); T_min = {Tmin[0]:.3f}/{Tmin[1]:.1e}/{Tmin[2]:.3f}; "
          f"reference T_min = {REF.t_min} <-> Δg = {dg_ref['over_coupled_smaller_gap']:+.1f} / {dg_ref['under_coupled_larger_gap']:+.1f} nm")
res["point_coupler"] = cap

# SAX ring circuit (brief snippet) at three gap errors: the notch depth is the visible consequence
ring, _ = sax.circuit(netlist={"instances": {"c": "coupler", "r": "waveguide"},
                               "connections": {"c,out1": "r,in0", "r,out0": "c,in1"},
                               "ports": {"in": "c,in0", "out": "c,out0"}},
                      models={"coupler": coupler, "waveguide": waveguide})
wl = np.linspace(1.298, 1.322, 24001)                              # covers > 2 FSR (10.3 nm)
S0 = ring(wl=wl, c={"coupling": REF.kappa2})
T0 = np.abs(np.asarray(S0[("in", "out")])) ** 2
from scipy.signal import find_peaks
pk, _ = find_peaks(-T0, prominence=0.5)
lam_r = wl[pk[np.argmin(np.abs(wl[pk] - 1.31))]]                       # resonance nearest 1310 nm
fsr_nm = float(np.mean(np.diff(wl[pk])) * 1e3) if len(pk) > 1 else None
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axs[0]
dg_fine = np.linspace(-0.020, 0.020, 401)
for i, (name, gam, lab) in enumerate((("textbook_neff2p5", g_text, "γ from n_eff = 2.5 (1/γ = 102 nm, textbook)"),
                                      ("slab2d_neff2p99", g_slab, "γ from the 2-D slab (1/γ = 80 nm)"))):
    Leff = sc.point_coupler_length(R, gam); kL0 = np.arcsin(np.sqrt(REF.kappa2))
    k2 = np.sin(kL0 * np.exp(-gam * dg_fine)) ** 2
    tt = np.sqrt(1 - k2); Tm = ((tt - A_RT) / (1 - tt * A_RT)) ** 2
    ax.plot(dg_fine * 1e3, -10 * np.log10(np.maximum(Tm, 1e-6)), color=SERIES[i], label=lab)
ax.axvline(0, color=PALETTE["muted"], lw=0.8); ax.axhline(18, color=PALETTE["muted"], lw=0.8, ls=":")
ax.text(19.5, 18.6, "18 dB = reference T_min 0.016", fontsize=7.5, color=PALETTE["ink2"], ha="right")   # right end: clear of both curves
dgr = cap["textbook_neff2p5"]["gap_error_nm_for_ref_Tmin"]
ax.plot([dgr["over_coupled_smaller_gap"], dgr["under_coupled_larger_gap"]], [-10 * np.log10(REF.t_min)] * 2, "o", ms=5,
        color=SERIES[3], label=f"computed: T_min = {REF.t_min} at Δg = {dgr['over_coupled_smaller_gap']:+.1f} / {dgr['under_coupled_larger_gap']:+.1f} nm (textbook γ)")
ax.set_xlabel("coupler gap error Δg (nm)"); ax.set_ylabel("on-resonance extinction  −10 log₁₀ T_min  (dB)")
ax.set_title(f"Critical coupling t = a needs the gap within a few nm:\nextinction vs gap error, a = {A_RT}, κ² = 0.107 nominal", fontsize=9.5)
ax.legend(fontsize=7.5, loc="lower center"); ax.set_ylim(0, 45)
ax = axs[1]
spec = {}
win = np.abs(wl - lam_r) < 0.0012
for i, d in enumerate((-0.010, 0.0, +0.010)):
    k2 = float(np.sin(kL0 * np.exp(-g_text * d)) ** 2)
    Sr = ring(wl=wl, c={"coupling": k2})
    T = np.abs(np.asarray(Sr[("in", "out")])) ** 2
    half = (T <= 0.5 * (T[win].min() + 1.0)) & win                  # half-depth width of this notch
    fwhm = float(np.ptp(wl[half]) * 1e6) if half.any() else None
    lab = f"Δg = {d*1e3:+.0f} nm: κ² = {k2:.3f}, t = {np.sqrt(1-k2):.3f}, T_min = {T[win].min():.3f}, FWHM {fwhm:.0f} pm"
    ax.plot((wl[win] - lam_r) * 1e6, 10 * np.log10(T[win]), color=SERIES[i], label=lab)
    spec[f"{int(d*1e3):+d}"] = dict(kappa2=k2, T_min=float(T[win].min()), T_min_dB=float(10 * np.log10(T[win].min())), fwhm_pm=fwhm)
spec["resonance_nearest_1310_nm"] = float(lam_r * 1e3); spec["fsr_nm"] = fsr_nm
spec["fsr_expected_nm"] = float(LAM ** 2 / (REF.ng * REF.round_trip_um) * 1e3)
spec["a_used"] = float(A_RT); spec["loss_db_cm_used"] = float(LOSS_DB_CM)
spec["fwhm_expected_pm_analytic"] = float(1e6 * LAM ** 2 / (np.pi * REF.ng * REF.round_trip_um) * (1 - A_RT * REF.t_coupler) / np.sqrt(A_RT * REF.t_coupler))
ax.axhline(10 * np.log10(REF.t_min), color=PALETTE["muted"], lw=0.8, ls=":")
ax.text(-1150, 10 * np.log10(REF.t_min) + 0.8, "reference T_min = 0.016 (−18 dB)", fontsize=7.5, color=PALETTE["ink2"])
ax.set_xlabel(f"λ − λ_r (pm),  λ_r = {lam_r*1e3:.1f} nm"); ax.set_ylabel("through-port transmission (dB)")
ax.set_title(f"SAX ring (L = 39.6 µm, a = {A_RT} via {LOSS_DB_CM:.1f} dB/cm, FSR {fsr_nm:.1f} nm):\na ±10 nm gap error changes the notch depth a lot\nand the loaded width by ~10 %, but not the position", fontsize=9.5)
ax.legend(fontsize=7, loc="lower right"); ax.set_ylim(-45, 1)
fig.tight_layout(); fig.savefig(OUT / FIGS[2]); plt.close(fig)
res["ring_spectra_textbook_gamma"] = spec
print(f"[sax] ring notch at {lam_r*1e3:.1f} nm, FSR {fsr_nm:.2f} nm (expect {spec['fsr_expected_nm']:.2f}); T_min at Δg = -10/0/+10 nm = "
      f"{spec['-10']['T_min']:.3f} / {spec['+0']['T_min']:.2e} / {spec['+10']['T_min']:.3f}; FWHM {spec['+0']['fwhm_pm']:.0f} pm (ref {REF.fwhm_pm:.0f})")

res["versions"] = dict(sax=sax.__version__, jax=__import__("jax").__version__, numpy=np.__version__)
res["figures"] = FIGS
(OUT / "sax_results.json").write_text(json.dumps(res, indent=2))
print("[sax] wrote", OUT / "sax_results.json")
