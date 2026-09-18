"""05_pulse_dispersion: how a pulse is delayed and distorted (notes section 12).

Propagate (a) a short Gaussian pulse and (b) a 53.125 Gbaud NRZ PRBS through
L = 2 km of bulk fused silica (material dispersion only, Malitson Sellmeier fit)
at 1310 nm and 1550 nm using the FFT method:

    A_out(t) = IFFT{ FFT{A_in(t)} * exp(-j [beta(omega_c + Omega) - beta0 - beta1 Omega] L) }

i.e. the notes' frequency integral with the exact beta(omega), written in the
retarded frame t' = t - beta1 L (the pure delay t_g = beta1 L is measured
separately in the absolute frame with a long pulse).

Run:  cd experiments/05_pulse_dispersion && ../../.venv/bin/python run.py
"""
import sys, pathlib, json, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
import scipy, scipy.signal as sig, scipy.optimize
import sympy

import silica
from silica import C0

T_START = time.time()
FFMPEG = "/opt/homebrew/bin/ffmpeg"
plt.rcParams["animation.ffmpeg_path"] = FFMPEG

# start from scratch: every file in out/ is produced by this script, so stale ones are removed first
for _old in OUT.iterdir():
    if _old.is_file() and _old.suffix in (".png", ".mp4", ".json", ".txt", ".csv"):
        _old.unlink()


def ffmpeg_version() -> str:
    import subprocess
    try:
        first = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        return first.split()[2]
    except Exception:
        return "unknown"

# ------------------------------------------------------------------ parameters
L_LINK = 2000.0                      # m   (a 2 km data-centre link)
LAMS = {"1310 nm": 1310e-9, "1550 nm": 1550e-9}
COL = {"1310 nm": SERIES[0], "1550 nm": SERIES[1]}
BAUD = REF.baud                      # 53.125e9 symbols/s
T_BIT = 1 / BAUD                     # 18.82 ps
T0_GAUSS = 5e-12                     # Gaussian field half-width at 1/e (intensity FWHM = 2 sqrt(ln2) T0 = 8.3 ps)
TX_F3DB = 0.75 * BAUD                # transmitter Gaussian low-pass (electro-optic bandwidth)
RX_F3DB = 0.75 * BAUD                # receiver 4th-order Bessel low-pass
EXT_RATIO_DB = 10.0                  # NRZ extinction ratio P1/P0
PRBS_ORDER = 9                       # 511-bit maximal-length sequence (periodic -> FFT-friendly)
SPB = 32                             # samples per bit
ALPHA_CHIRP = 3.0                    # linewidth-enhancement factor of a directly modulated laser (secondary scenario);
                                     # field = sqrt(P) exp(j (alpha/2) ln P): instantaneous frequency shift (alpha/4pi) dlnP/dt,
                                     # so with alpha > 0 the rising edge is blue-shifted (DML transient chirp). 0 = chirp-free.

results = {}
log_lines = []


def log(s=""):
    print(s); log_lines.append(s)


def pct(a, b):
    """Percent agreement of a with expectation b."""
    return 100.0 * (1 - abs(a - b) / abs(b)) if b != 0 else float("nan")


# ------------------------------------------------------------------ 1. material numbers
log("== 1. Bulk silica material dispersion (Malitson Sellmeier, notes 9-11) ==")
log(silica.symbolic_summary())
mat = {}
for name, lam in LAMS.items():
    w = silica.omega_of_lam(lam)
    b1, b2, b3 = silica.beta1(w), silica.beta2(w), silica.beta3(w)
    Dv = silica.D(lam)
    b2_formula = -lam**2 * Dv / (2 * np.pi * C0)
    mat[name] = dict(lam=lam, omega=w, n=float(silica.n(lam)), ng=float(silica.n_g(lam)),
                     D_si=float(Dv), D_ps_nm_km=float(Dv * 1e6),
                     beta1=b1, beta2=b2, beta3=b3, beta2_formula=b2_formula)
    log(f"{name}: n = {mat[name]['n']:.5f}, n_g = {mat[name]['ng']:.5f}, D = {Dv*1e6:+.3f} ps/(nm km), "
        f"beta1 = {b1*1e12:.4f} ps/m (= n_g/c: {mat[name]['ng']/C0*1e12:.4f}), "
        f"beta2 = {b2*1e27:+.3f} ps^2/km (formula -lam^2 D/2pi c: {b2_formula*1e27:+.3f}), "
        f"beta3 = {b3*1e39:+.4f} ps^3/km")
zdw = silica.zero_dispersion_wavelength()
log(f"zero-dispersion wavelength of bulk silica = {zdw*1e9:.1f} nm (fibre: ~1310 nm once waveguide dispersion is added)")
results["material"] = {
    "n_1310": mat["1310 nm"]["n"], "n_1310_expected_notes": 1.4468,
    "ng_1310": mat["1310 nm"]["ng"], "ng_1550": mat["1550 nm"]["ng"],
    "D_1310_ps_nm_km": mat["1310 nm"]["D_ps_nm_km"], "D_1550_ps_nm_km": mat["1550 nm"]["D_ps_nm_km"],
    "D_1550_expected_bulk": "20-22", "beta2_1310_ps2_km": mat["1310 nm"]["beta2"] * 1e27,
    "beta2_1550_ps2_km": mat["1550 nm"]["beta2"] * 1e27,
    "beta2_vs_formula_agreement_pct": pct(mat["1550 nm"]["beta2"], mat["1550 nm"]["beta2_formula"]),
    "beta3_1310_ps3_km": mat["1310 nm"]["beta3"] * 1e39, "beta3_1550_ps3_km": mat["1550 nm"]["beta3"] * 1e39,
    "zero_dispersion_wavelength_nm": zdw * 1e9, "zdw_expected_nm": "~1270 (bulk)",
}

# figure: n, n_g, D over 1.0-1.7 um
lam_ax = np.linspace(1.0e-6, 1.7e-6, 600)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].plot(lam_ax * 1e9, silica.n(lam_ax), color=SERIES[0], label="n (phase index)")
axs[0].plot(lam_ax * 1e9, silica.n_g(lam_ax), color=SERIES[1], label="n_g = n − λ dn/dλ (group index)")
axs[0].set_ylabel("index"); axs[0].legend()
axs[0].set_title("n falls with λ while n_g has a minimum\nat the zero-dispersion point", fontsize=9)
axs[1].plot(lam_ax * 1e9, silica.D_ps_nm_km(lam_ax), color=SERIES[2])
axs[1].axhline(0, color=PALETTE["muted"], lw=0.8)
axs[1].set_ylabel("D = −(λ/c) d²n/dλ²  [ps/(nm·km)]")
axs[1].set_title(f"D crosses zero at {zdw*1e9:.0f} nm (bulk silica);\n1310 nm sits just above it, 1550 nm far above", fontsize=9)
b2_ax = np.array([silica.beta2(silica.omega_of_lam(l)) for l in lam_ax]) * 1e27
axs[2].plot(lam_ax * 1e9, b2_ax, color=SERIES[3])
axs[2].axhline(0, color=PALETTE["muted"], lw=0.8)
axs[2].set_ylabel("β₂ = d²β/dω²  [ps²/km]")
axs[2].set_title(f"β₂ = −λ²D/(2πc): {mat['1550 nm']['beta2']/mat['1310 nm']['beta2']:.1f}× larger\nat 1550 nm than at 1310 nm", fontsize=9)
BAND = {"1310 nm": "O-band carrier", "1550 nm": "C-band carrier"}
for ax in axs:
    ax.set_xlabel("vacuum wavelength λ [nm]")
    for name, lam in LAMS.items():
        ax.axvline(lam * 1e9, color=COL[name], ls="--", lw=1, alpha=0.7, label=f"{name} ({BAND[name]})")
# annotation text is placed in empty regions of the panel (D runs from -40 to +31 over this range) so it never sits on the curve
axs[1].annotate(f"D(1310) = {mat['1310 nm']['D_ps_nm_km']:.1f}", (1310, mat['1310 nm']['D_ps_nm_km']), xytext=(1340, -14), fontsize=8, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
axs[1].annotate(f"D(1550) = {mat['1550 nm']['D_ps_nm_km']:.1f}", (1550, mat['1550 nm']['D_ps_nm_km']), xytext=(1330, 30), fontsize=8, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
# same for beta2 in the third panel (beta2 falls from +21 to -48 top-left to bottom-right, so above-left and below-right are empty)
axs[2].annotate(f"β₂(1310) = {mat['1310 nm']['beta2']*1e27:.1f}", (1310, mat['1310 nm']['beta2'] * 1e27), xytext=(1340, 12), fontsize=8, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
axs[2].annotate(f"β₂(1550) = {mat['1550 nm']['beta2']*1e27:.1f}", (1550, mat['1550 nm']['beta2'] * 1e27), xytext=(1390, -42), fontsize=8, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
# every panel gets a legend so the dashed marker lines are named on each of them, not only on the first
axs[0].legend(fontsize=8); axs[1].legend(fontsize=8, loc="lower right"); axs[2].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "silica_dispersion.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ propagation helpers
def propagate_retarded(A_in, dt, omega_c, z):
    """FFT propagation in the retarded frame t' = t - beta1(omega_c) z using the EXACT beta(omega).

    H(Omega) = exp(-j [beta(omega_c+Omega) - beta0 - beta1 Omega] z): the constant carrier
    phase and the pure delay are removed, everything else (beta2, beta3, ...) is kept.
    """
    N = len(A_in)
    Om = 2 * np.pi * np.fft.fftfreq(N, dt)
    b0 = silica.beta(omega_c); b1 = silica.beta1(omega_c)
    phase = (silica.beta(omega_c + Om) - b0 - b1 * Om) * z
    return np.fft.ifft(np.fft.fft(A_in) * np.exp(-1j * phase))


def propagate_taylor2(A_in, dt, omega_c, z):
    """Same but with only the beta2 term (notes 12 quadratic truncation)."""
    N = len(A_in)
    Om = 2 * np.pi * np.fft.fftfreq(N, dt)
    return np.fft.ifft(np.fft.fft(A_in) * np.exp(-1j * 0.5 * silica.beta2(omega_c) * Om**2 * z))


def propagate_absolute(A_in, dt, omega_c, z):
    """Absolute-time propagation exp(-j beta(omega) z): the output appears delayed by t_g = beta1 z."""
    N = len(A_in)
    Om = 2 * np.pi * np.fft.fftfreq(N, dt)
    phase = np.mod(silica.beta(omega_c + Om) * z, 2 * np.pi)     # wrap the ~1e10 rad phase explicitly
    return np.fft.ifft(np.fft.fft(A_in) * np.exp(-1j * phase))


def fwhm(t, P):
    """Full width at half maximum of a single-peaked P(t) by linear interpolation."""
    half = P.max() / 2
    idx = np.where(P >= half)[0]
    i0, i1 = idx[0], idx[-1]
    tl = np.interp(half, [P[i0 - 1], P[i0]], [t[i0 - 1], t[i0]])
    tr = np.interp(half, [P[i1 + 1], P[i1]], [t[i1 + 1], t[i1]])
    return tr - tl


def rms_width(t, P):
    P = P / np.trapezoid(P, t)
    m = np.trapezoid(t * P, t)
    return np.sqrt(np.trapezoid((t - m) ** 2 * P, t))


def inst_freq_shift(A, dt):
    """Instantaneous frequency deviation from the carrier, (1/2pi) d(arg A)/dt, in Hz."""
    ph = np.unwrap(np.angle(A))
    return np.gradient(ph, dt) / (2 * np.pi)


# ------------------------------------------------------------------ 2. Gaussian pulse
log("\n== 2. Gaussian pulse, T0 = %.1f ps, L = %.0f m (notes 12) ==" % (T0_GAUSS * 1e12, L_LINK))
N_G = 2 ** 14; dt_g = 0.05e-12
t_g = (np.arange(N_G) - N_G // 2) * dt_g
A0 = np.exp(-t_g**2 / (2 * T0_GAUSS**2)).astype(complex)      # |A|^2 = exp(-t^2/T0^2)
P_in = np.abs(A0) ** 2
fwhm_in = fwhm(t_g, P_in)
z_grid = np.linspace(0, L_LINK, 81)
gauss = {}
for name, m in mat.items():
    wc = m["omega"]
    LD = T0_GAUSS**2 / abs(m["beta2"])                          # dispersion length
    widths_sim, widths_an = [], []
    for z in z_grid:
        Pz = np.abs(propagate_retarded(A0, dt_g, wc, z)) ** 2
        widths_sim.append(fwhm(t_g, Pz))
        widths_an.append(fwhm_in * np.sqrt(1 + (z / LD) ** 2))
    A_out = propagate_retarded(A0, dt_g, wc, L_LINK)
    A_out_t2 = propagate_taylor2(A0, dt_g, wc, L_LINK)
    P_out = np.abs(A_out) ** 2
    gauss[name] = dict(LD=LD, widths_sim=np.array(widths_sim), widths_an=np.array(widths_an),
                       A_out=A_out, P_out=P_out, fwhm_out=fwhm(t_g, P_out), fwhm_an=widths_an[-1],
                       peak_out=P_out.max(), peak_an=1 / np.sqrt(1 + (L_LINK / LD) ** 2),
                       taylor2_max_diff=float(np.max(np.abs(np.abs(A_out_t2) ** 2 - P_out))),
                       energy_ratio=float(np.trapezoid(P_out, t_g) / np.trapezoid(P_in, t_g)),
                       chirp=inst_freq_shift(A_out, dt_g))
    g = gauss[name]
    log(f"{name}: L_D = T0^2/|beta2| = {LD:.0f} m; FWHM in {fwhm_in*1e12:.2f} ps -> out {g['fwhm_out']*1e12:.2f} ps "
        f"(analytic T0 sqrt(1+(L/L_D)^2): {g['fwhm_an']*1e12:.2f} ps, agreement {pct(g['fwhm_out'], g['fwhm_an']):.2f} %); "
        f"peak {g['peak_out']:.3f} (analytic {g['peak_an']:.3f}); energy conserved to {g['energy_ratio']:.6f}; "
        f"max |P_exact - P_beta2only| = {g['taylor2_max_diff']:.2e} (beta3 negligible)")
results["gaussian"] = {
    "T0_ps": T0_GAUSS * 1e12, "fwhm_in_ps": fwhm_in * 1e12, "L_m": L_LINK,
    **{f"{k}_{name[:4]}": v for name, g in gauss.items()
       for k, v in (("L_D_m", g["LD"]), ("fwhm_out_ps", g["fwhm_out"] * 1e12), ("fwhm_out_analytic_ps", g["fwhm_an"] * 1e12),
                    ("fwhm_agreement_pct", pct(g["fwhm_out"], g["fwhm_an"])), ("peak_out", g["peak_out"]),
                    ("peak_analytic", g["peak_an"]), ("peak_agreement_pct", pct(g["peak_out"], g["peak_an"])),
                    ("energy_ratio", g["energy_ratio"]), ("beta3_effect_max_dP", g["taylor2_max_diff"]))},
}

# figure: in/out envelopes + chirp + width vs z
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
axs[0].plot(t_g * 1e12, P_in, color=PALETTE["ink2"], ls="--", label="input (z = 0)")
for name, g in gauss.items():
    axs[0].plot(t_g * 1e12, g["P_out"], color=COL[name], label=f"output, {name}")
axs[0].set_xlim(-40, 40); axs[0].set_xlabel("retarded time t − β₁L [ps]"); axs[0].set_ylabel("|A|² (normalised)")
axs[0].set_title(f"After 2 km the 1550 nm pulse is {gauss['1550 nm']['fwhm_out']/fwhm_in:.1f}× wider,\nthe 1310 nm pulse barely changes", fontsize=9)
axs[0].legend()
for name, g in gauss.items():
    mask = g["P_out"] > 0.02 * g["P_out"].max()
    axs[1].plot(t_g[mask] * 1e12, g["chirp"][mask] / 1e9, color=COL[name], label=name)
axs[1].axhline(0, color=PALETTE["muted"], lw=0.8)
axs[1].set_xlim(-40, 40); axs[1].set_xlabel("retarded time t − β₁L [ps]"); axs[1].set_ylabel("instantaneous frequency − f_c [GHz]")
axs[1].set_title("β₂ < 0 (anomalous): the blue (higher-f) part runs ahead,\nthe red part lags behind: a linear chirp", fontsize=9)
axs[1].legend()
for name, g in gauss.items():
    axs[2].plot(z_grid / 1e3, g["widths_sim"] * 1e12, color=COL[name], label=f"FFT simulation, {name}")
    axs[2].plot(z_grid / 1e3, g["widths_an"] * 1e12, color=COL[name], ls=":", lw=1.5, label=f"T₀√(1+(z/L_D)²), {name}")
axs[2].set_xlabel("distance z [km]"); axs[2].set_ylabel("intensity FWHM [ps]")
axs[2].set_title("Width vs distance follows the Gaussian formula\nwith L_D = T₀²/|β₂|", fontsize=9)
axs[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "gaussian_broadening.png"); plt.close(fig)


# ------------------------------------------------------------------ 3. absolute group delay with a long pulse
log("\n== 3. Group delay t_g = beta1 L measured in the absolute frame (long pulse) ==")
N_D = 2 ** 18; dt_d = 0.05e-9                              # 13.1 us window, 50 ps resolution
t_d = np.arange(N_D) * dt_d
T0_long = 2e-9
A_long = np.exp(-(t_d - 1e-6) ** 2 / (2 * T0_long**2)).astype(complex)   # launched centred at 1 us
delay = {}
for name, m in mat.items():
    A_out = propagate_absolute(A_long, dt_d, m["omega"], L_LINK)
    P = np.abs(A_out) ** 2
    Pn = P / np.trapezoid(P, t_d)
    t_peak = np.trapezoid(t_d * Pn, t_d) - 1e-6                  # centroid, minus launch time
    tg_an = m["beta1"] * L_LINK
    tphase = m["n"] * L_LINK / C0
    delay[name] = dict(t_meas=t_peak, t_g=tg_an, t_phase=tphase, P=P)
    log(f"{name}: measured arrival {t_peak*1e6:.5f} us; t_g = beta1 L = n_g L/c = {tg_an*1e6:.5f} us "
        f"(agreement {pct(t_peak, tg_an):.4f} %); phase delay n L/c = {tphase*1e6:.5f} us "
        f"(differs by {(tg_an-tphase)*1e9:.1f} ns: it is n_g, not n, that sets the delay)")
dt_1550_1310 = delay["1550 nm"]["t_meas"] - delay["1310 nm"]["t_meas"]
dt_an = (mat["1550 nm"]["beta1"] - mat["1310 nm"]["beta1"]) * L_LINK
log(f"1550 arrives {dt_1550_1310*1e9:.2f} ns after 1310 (analytic (n_g1550 - n_g1310) L/c = {dt_an*1e9:.2f} ns)")
results["group_delay"] = {
    **{f"t_g_meas_us_{n[:4]}": d["t_meas"] * 1e6 for n, d in delay.items()},
    **{f"t_g_analytic_us_{n[:4]}": d["t_g"] * 1e6 for n, d in delay.items()},
    **{f"t_phase_nL_over_c_us_{n[:4]}": d["t_phase"] * 1e6 for n, d in delay.items()},
    "t_g_agreement_pct_1310": pct(delay["1310 nm"]["t_meas"], delay["1310 nm"]["t_g"]),
    "t_g_agreement_pct_1550": pct(delay["1550 nm"]["t_meas"], delay["1550 nm"]["t_g"]),
    "arrival_diff_1550_minus_1310_ns": dt_1550_1310 * 1e9, "arrival_diff_analytic_ns": dt_an * 1e9,
}
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
axs[0].plot(t_d * 1e6, np.abs(A_long) ** 2, color=PALETTE["ink2"], ls="--", label="input (launched at 1 µs)")
for name, d in delay.items():
    axs[0].plot(t_d * 1e6, d["P"] / d["P"].max(), color=COL[name], label=f"output, {name}")
axs[0].axvline((1e-6 + delay["1310 nm"]["t_phase"]) * 1e6, color=PALETTE["muted"], ls=":", label="n·L/c (phase delay, wrong)")
axs[0].set_xlabel("absolute time t [µs]"); axs[0].set_ylabel("|A|² (normalised)")
axs[0].set_title(f"A 2 ns pulse arrives after t_g = n_g L/c ≈ {delay['1310 nm']['t_g']*1e6:.2f} µs, not n L/c"); axs[0].legend(fontsize=8)
zoom_c = 1e-6 + delay["1310 nm"]["t_g"]
for name, d in delay.items():
    axs[1].plot((t_d - zoom_c) * 1e9, d["P"] / d["P"].max(), color=COL[name], label=f"{name}")
    axs[1].axvline((d["t_g"] - delay["1310 nm"]["t_g"]) * 1e9, color=COL[name], ls=":", lw=1)
axs[1].axvline((delay["1310 nm"]["t_phase"] - delay["1310 nm"]["t_g"]) * 1e9, color=PALETTE["muted"], ls=":", label="n·L/c at 1310")
axs[1].set_xlim(-120, 15); axs[1].set_xlabel("time − t_g(1310) [ns]"); axs[1].set_ylabel("|A|² (normalised)")
axs[1].set_title(f"Zoom: 1550 nm arrives {dt_1550_1310*1e9:.1f} ns later (n_g larger);\n"
                 f"n·L/c is {(delay['1310 nm']['t_g']-delay['1310 nm']['t_phase'])*1e9:.0f} ns early")
axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "group_delay.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 4. NRZ PRBS link
log("\n== 4. 53.125 Gbaud NRZ PRBS through 2 km (eye diagrams) ==")
bits = sig.max_len_seq(PRBS_ORDER)[0]                     # 511 bits, periodic
N_BITS = len(bits)
N_S = N_BITS * SPB; dt_s = T_BIT / SPB
t_s = np.arange(N_S) * dt_s
f_s = np.fft.fftfreq(N_S, dt_s)
P1, P0 = 1.0, 10 ** (-EXT_RATIO_DB / 10)
drive = np.where(np.repeat(bits, SPB) == 1, P1, P0).astype(float)
# transmitter: Gaussian low-pass (3 dB at TX_F3DB) applied to the optical power waveform -> chirp-free field sqrt(P)
sigma_tx = TX_F3DB / np.sqrt(np.log(2))
H_tx = np.exp(-0.5 * (f_s / sigma_tx) ** 2)
P_tx = np.real(np.fft.ifft(np.fft.fft(drive) * H_tx))
P_tx = np.clip(P_tx, 0, None)
E_tx = np.sqrt(P_tx).astype(complex)
# receiver: 4th-order Bessel low-pass, applied in the frequency domain to the detected power
b_rx, a_rx = sig.bessel(4, 2 * np.pi * RX_F3DB, "low", analog=True, norm="mag")
_, H_rx = sig.freqs(b_rx, a_rx, worN=2 * np.pi * f_s)
# the Bessel filter is causal, so it delays everything by its (flat) group delay tau_rx = -d(arg H_rx)/d(omega) at
# omega -> 0.  The eye metric absorbs this through the best sampling phase; the waveform figure removes it explicitly
# so the dashed "after Rx filter" trace is not mistaken for a dispersion delay.
_f_small = 1e7
_, _H_small = sig.freqs(b_rx, a_rx, worN=2 * np.pi * np.array([0.0, _f_small]))
TAU_RX = float(-(np.angle(_H_small[1]) - np.angle(_H_small[0])) / (2 * np.pi * _f_small))


def receive(E):
    return np.real(np.fft.ifft(np.fft.fft(np.abs(E) ** 2) * H_rx))


def eye_fold(y, n_ui=2):
    """Fold a periodic waveform into traces of n_ui unit intervals (one trace per bit)."""
    traces = []
    for k in range(N_BITS):
        idx = (np.arange(n_ui * SPB) + k * SPB) % N_S
        traces.append(y[idx])
    return np.array(traces)


def eye_metrics(y):
    """Eye height: max over sampling phase of (min of the 1-level samples - max of the 0-level samples).

    Uses the known bit pattern; the best sampling phase absorbs filter delays.
    """
    best = -np.inf; best_ph = 0
    for ph in range(SPB):
        samples = y[(np.arange(N_BITS) * SPB + ph) % N_S]
        ones, zeros = samples[bits == 1], samples[bits == 0]
        h = ones.min() - zeros.max()
        if h > best:
            best, best_ph = h, ph
    samples = y[(np.arange(N_BITS) * SPB + best_ph) % N_S]
    return dict(height=best, phase=best_ph, mean1=samples[bits == 1].mean(), mean0=samples[bits == 0].mean())


y_b2b = receive(E_tx)
m_b2b = eye_metrics(y_b2b)
log(f"bit period T_b = {T_BIT*1e12:.2f} ps; Tx Gaussian LPF {TX_F3DB/1e9:.1f} GHz; Rx Bessel-4 {RX_F3DB/1e9:.1f} GHz; ER {EXT_RATIO_DB} dB; PRBS-{PRBS_ORDER}")
log(f"back-to-back eye height = {m_b2b['height']:.3f} (of P1 = 1); Rx Bessel filter DC group delay {TAU_RX*1e12:.2f} ps "
    f"(= {TAU_RX/T_BIT:.2f} UI; removed from the 'after Rx filter' trace in nrz_waveforms.png only)")
nrz = {}
for name, m in mat.items():
    E_out = propagate_retarded(E_tx, dt_s, m["omega"], L_LINK)
    y = receive(E_out)
    met = eye_metrics(y)
    # delay spread across the double-sided optical bandwidth of the signal (+/- TX_F3DB around the carrier)
    dOm = 2 * np.pi * 2 * TX_F3DB
    spread = abs(m["beta2"]) * L_LINK * dOm
    dlam = 2 * TX_F3DB * m["lam"] ** 2 / C0
    spread_DL = abs(m["D_si"]) * L_LINK * dlam
    # phase curvature across the band, a dimensionless dispersion strength
    phi_band = 0.5 * abs(m["beta2"]) * L_LINK * (2 * np.pi * TX_F3DB) ** 2
    nrz[name] = dict(E_out=E_out, y=y, met=met, spread=spread, spread_DL=spread_DL, dlam=dlam, phi=phi_band,
                     opening=met["height"] / m_b2b["height"])
    log(f"{name}: delay spread |beta2| L 2pi(2 f3dB) = {spread*1e12:.1f} ps (= D L dlam with dlam = {dlam*1e9:.3f} nm: {spread_DL*1e12:.1f} ps); "
        f"|beta2| L (2pi f3dB)^2/2 = {phi_band:.2f} rad; eye height {met['height']:.3f} -> opening {100*nrz[name]['opening']:.1f} % of back-to-back")
# Secondary scenario: the same NRZ power waveform from a chirped (directly modulated) laser.
# E = sqrt(P) exp(j (alpha/2) ln P)  ->  instantaneous frequency shift (alpha/4pi) d ln P/dt (Hz):
# the rising edge is blue-shifted, the falling edge red-shifted (alpha > 0).  This widens the optical
# spectrum without changing the power waveform, so the receiver sees the same back-to-back eye.
E_tx_chirp = np.sqrt(P_tx) * np.exp(1j * 0.5 * ALPHA_CHIRP * np.log(np.clip(P_tx, 1e-6, None)))
rms_bw = lambda E: np.sqrt(np.sum(f_s**2 * np.abs(np.fft.fft(E))**2) / np.sum(np.abs(np.fft.fft(E))**2))
log(f"chirp-free Tx: rms optical bandwidth {rms_bw(E_tx)/1e9:.1f} GHz; DML-like Tx (alpha = {ALPHA_CHIRP}): {rms_bw(E_tx_chirp)/1e9:.1f} GHz "
    f"(same power waveform, back-to-back eye {eye_metrics(receive(E_tx_chirp))['height']:.3f})")
nrz_chirp = {}
for name, m in mat.items():
    E_out = propagate_retarded(E_tx_chirp, dt_s, m["omega"], L_LINK)
    y = receive(E_out); met = eye_metrics(y)
    nrz_chirp[name] = dict(E_out=E_out, y=y, met=met, opening=met["height"] / m_b2b["height"])
    log(f"{name}, alpha = {ALPHA_CHIRP}: eye height {met['height']:+.3f} -> opening {100*nrz_chirp[name]['opening']:.1f} % of back-to-back"
        + ("  (CLOSED: the worst 1 sample is below the best 0 sample)" if met["height"] <= 0 else ""))
results["nrz"] = {
    "baud_GHz": BAUD / 1e9, "bit_period_ps": T_BIT * 1e12, "tx_f3dB_GHz": TX_F3DB / 1e9, "rx_f3dB_GHz": RX_F3DB / 1e9,
    "extinction_ratio_dB": EXT_RATIO_DB, "prbs_order": PRBS_ORDER, "eye_height_b2b": m_b2b["height"],
    "rx_filter_dc_group_delay_ps": TAU_RX * 1e12, "rx_filter_dc_group_delay_UI": TAU_RX / T_BIT,
    **{f"delay_spread_ps_{n[:4]}": d["spread"] * 1e12 for n, d in nrz.items()},
    **{f"optical_bandwidth_nm_{n[:4]}": d["dlam"] * 1e9 for n, d in nrz.items()},
    **{f"band_phase_curvature_rad_{n[:4]}": d["phi"] for n, d in nrz.items()},
    "delay_spread_1550_expected_brief_ps": 27.0,
    "delay_spread_1550_vs_brief_agreement_pct": pct(nrz["1550 nm"]["spread"] * 1e12, 27.0),
    **{f"eye_height_{n[:4]}": d["met"]["height"] for n, d in nrz.items()},
    **{f"eye_opening_fraction_{n[:4]}": d["opening"] for n, d in nrz.items()},
    "chirp_alpha": ALPHA_CHIRP,
    "rms_optical_bandwidth_GHz_chirp_free": float(rms_bw(E_tx) / 1e9),
    "rms_optical_bandwidth_GHz_chirped": float(rms_bw(E_tx_chirp) / 1e9),
    **{f"eye_height_chirped_{n[:4]}": d["met"]["height"] for n, d in nrz_chirp.items()},
    **{f"eye_opening_fraction_chirped_{n[:4]}": d["opening"] for n, d in nrz_chirp.items()},
}

# eye vs length sweep (chirp-free and chirped)
L_sweep = np.linspace(0, 10e3, 41)
sweep = {}


def eye_opening_at(E_launch, omega_c, Lz):
    """Eye opening (fraction of back-to-back) of a launched field after Lz metres."""
    return eye_metrics(receive(propagate_retarded(E_launch, dt_s, omega_c, Lz)))["height"] / m_b2b["height"]


REACH_XTOL_M = 1.0


def reach_50(op, E_launch, omega_c):
    """Length at which the eye opening first drops below 50 % of back-to-back (None if never within the sweep).

    The 250 m sweep grid only brackets the crossing; the exact length is then found with
    scipy.optimize.brentq on the eye-opening function itself (to REACH_XTOL_M), so the
    quoted reach is not limited by the grid spacing.
    """
    below = np.where(op < 0.5)[0]
    if not len(below):
        return None
    i = below[0]
    return float(scipy.optimize.brentq(lambda Lz: eye_opening_at(E_launch, omega_c, Lz) - 0.5,
                                       L_sweep[i - 1], L_sweep[i], xtol=REACH_XTOL_M))


for label, E_launch in (("", E_tx), (" chirped", E_tx_chirp)):
    for name, m in mat.items():
        op = np.array([eye_opening_at(E_launch, m["omega"], Lz) for Lz in L_sweep])
        reach = reach_50(op, E_launch, m["omega"])
        sweep[name + label] = dict(op=op, reach=reach)
        log(f"{name}{label or ' chirp-free'}: eye opening falls to 50 % at L = "
            + (f"{reach:.0f} m (brentq between the {L_sweep[1]:.0f} m sweep points, tolerance {REACH_XTOL_M:.0f} m)"
               if reach is not None else f"> {L_sweep[-1]/1e3:.0f} km (never within the sweep)"))
results["nrz"].update({"reach_sweep_max_km": L_sweep[-1] / 1e3, "reach_sweep_step_m": float(L_sweep[1]),
                       "reach_root_tolerance_m": REACH_XTOL_M})
results["nrz"].update({f"reach_50pct_eye_km_{n[:4]}{'_chirped' if 'chirped' in n else ''}": (s["reach"] / 1e3 if s["reach"] is not None else None)
                       for n, s in sweep.items()})
reach_str = lambda key: (f"{sweep[key]['reach']/1e3:.2f} km" if sweep[key]["reach"] is not None else f"> {L_sweep[-1]/1e3:.0f} km")

# ---- numbers quoted in the README's "Checks" and "Experiments to try", computed here so they are on record
log("\n== 4b. Checks and 'experiments to try' numbers (written to results.json) ==")
# (i) how much of the NRZ field's spectral energy sits inside +/- f of the carrier, EXCLUDING the DC carrier
#     line (the unmodulated mean power, which carries no data and is delayed by exactly beta1 L).
S_full = np.abs(np.fft.fft(E_tx)) ** 2
S_nodc = S_full.copy(); S_nodc[0] = 0.0
frac_in = lambda f_half: float(S_nodc[np.abs(f_s) <= f_half].sum() / S_nodc.sum())
frac_in_with_dc = lambda f_half: float(S_full[np.abs(f_s) <= f_half].sum() / S_full.sum())
carrier_share = float(S_full[0] / S_full.sum())
energy_fracs = {f"{round(f/1e9):.0f}GHz": frac_in(f) for f in (20e9, 27e9, TX_F3DB)}
log(f"NRZ spectral energy excluding the carrier line (which holds {100*carrier_share:.1f} % of the total): "
    + ", ".join(f"{100*v:.1f} % within +/-{k[:-3]} GHz" for k, v in energy_fracs.items())
    + f"  (including the carrier line: {100*frac_in_with_dc(20e9):.1f} % / {100*frac_in_with_dc(27e9):.1f} % / {100*frac_in_with_dc(TX_F3DB):.1f} %)")
# (ii) the classic dispersion-limited reach B^2 |beta2| L <~ 0.25  ->  L_max = 0.25 / (B^2 |beta2|)
L_b2 = {name: 0.25 / (BAUD**2 * abs(m["beta2"])) for name, m in mat.items()}
log("B^2 |beta2| L = 0.25 reach: " + ", ".join(f"{n} {L/1e3:.2f} km" for n, L in L_b2.items())
    + f"  (half-eye reach from the sweep, chirp-free: 1310 nm {reach_str('1310 nm')}, 1550 nm {reach_str('1550 nm')})")
# (iii) experiment 1: chirp-free eye at 10 km (end of the sweep) and 1310 nm beyond it
op_10km = {name: float(sweep[name]["op"][-1]) for name in LAMS}
op_1310_far = {Lz: float(eye_opening_at(E_tx, mat["1310 nm"]["omega"], Lz)) for Lz in (20e3, 23e3, 30e3, 40e3)}
log(f"experiment 1: chirp-free eye at 10 km: 1310 nm {100*op_10km['1310 nm']:.0f} %, 1550 nm {100*max(op_10km['1550 nm'],0):.0f} %; "
    f"1310 nm chirp-free at " + ", ".join(f"{Lz/1e3:.0f} km {100*max(v,0):.0f} %" for Lz, v in op_1310_far.items()))
# (iv) experiment 2: a 1 ps Gaussian (L_D scales with T0^2)
T0_short = 1e-12
A0_short = np.exp(-t_g**2 / (2 * T0_short**2)).astype(complex)
short = {}
for name, m in mat.items():
    P_short = np.abs(propagate_retarded(A0_short, dt_g, m["omega"], L_LINK)) ** 2
    short[name] = dict(LD=T0_short**2 / abs(m["beta2"]), fwhm_in=fwhm(t_g, np.abs(A0_short) ** 2), fwhm_out=fwhm(t_g, P_short))
log("experiment 2: T0 = 1 ps: " + "; ".join(f"{n}: L_D = {s['LD']:.0f} m, FWHM {s['fwhm_in']*1e12:.2f} -> {s['fwhm_out']*1e12:.1f} ps after 2 km" for n, s in short.items()))
# (v) experiment 3: the opposite chirp sign (alpha = -3) pre-compensates anomalous dispersion at 1550 nm
E_tx_neg = np.sqrt(P_tx) * np.exp(-1j * 0.5 * ALPHA_CHIRP * np.log(np.clip(P_tx, 1e-6, None)))
op_neg = {Lz: float(eye_opening_at(E_tx_neg, mat["1550 nm"]["omega"], Lz)) for Lz in (0.5e3, 1e3, 2e3, 3e3, 4e3)}
log(f"experiment 3: alpha = {-ALPHA_CHIRP:.0f} at 1550 nm: " + ", ".join(f"{Lz/1e3:.1f} km {100*max(v,0):.0f} %" for Lz, v in op_neg.items()))
# (vi) experiment 4: a wider transmitter AND receiver (1.5 x baud instead of 0.75 x baud) at 2 km.
#      The back-to-back reference is re-taken with the wide filters, so the opening is a like-for-like fraction.
F_WIDE = 1.5 * BAUD
H_tx_w = np.exp(-0.5 * (f_s / (F_WIDE / np.sqrt(np.log(2)))) ** 2)
P_tx_w = np.clip(np.real(np.fft.ifft(np.fft.fft(drive) * H_tx_w)), 0, None)
E_tx_w = np.sqrt(P_tx_w).astype(complex)
b_w, a_w = sig.bessel(4, 2 * np.pi * F_WIDE, "low", analog=True, norm="mag")
_, H_rx_w = sig.freqs(b_w, a_w, worN=2 * np.pi * f_s)
receive_w = lambda E: np.real(np.fft.ifft(np.fft.fft(np.abs(E) ** 2) * H_rx_w))
h_b2b_w = eye_metrics(receive_w(E_tx_w))["height"]
S_w = np.abs(np.fft.fft(E_tx_w)) ** 2; S_w[0] = 0.0
wide = {}
for name, m in mat.items():
    h = eye_metrics(receive_w(propagate_retarded(E_tx_w, dt_s, m["omega"], L_LINK)))["height"]
    wide[name] = dict(height=h, opening=h / h_b2b_w, spread=abs(m["beta2"]) * L_LINK * 2 * np.pi * 2 * F_WIDE)
log(f"experiment 4: Tx = Rx = {F_WIDE/1e9:.1f} GHz (1.5 x baud): back-to-back eye height {h_b2b_w:.3f} (vs {m_b2b['height']:.3f} at 0.75 x baud); "
    f"rms optical bandwidth {rms_bw(E_tx_w)/1e9:.1f} GHz (vs {rms_bw(E_tx)/1e9:.1f}); "
    + "; ".join(f"{n}: 2 km eye {100*max(w['opening'],0):.1f} % of its own back-to-back, delay spread across +/-{F_WIDE/1e9:.0f} GHz {w['spread']*1e12:.1f} ps"
                for n, w in wide.items()))
# (vii) experiment 5: a carrier at the zero-dispersion wavelength: only beta3 is left
w_zdw = silica.omega_of_lam(zdw)
P_zdw = np.abs(propagate_retarded(A0, dt_g, w_zdw, L_LINK)) ** 2
P_zdw_10 = np.abs(propagate_retarded(A0, dt_g, w_zdw, 10e3)) ** 2
op_zdw_10km = float(eye_opening_at(E_tx, w_zdw, 10e3))
log(f"experiment 5: carrier at the ZDW {zdw*1e9:.1f} nm: beta2 = {silica.beta2(w_zdw)*1e27:.1e} ps^2/km, beta3 = {silica.beta3(w_zdw)*1e39:.3f} ps^3/km; "
    f"Gaussian FWHM {fwhm_in*1e12:.4f} -> {fwhm(t_g, P_zdw)*1e12:.4f} ps at 2 km, {fwhm(t_g, P_zdw_10)*1e12:.4f} ps at 10 km; chirp-free eye at 10 km {100*op_zdw_10km:.1f} %")
results["checks"] = {
    "spectral_energy_fraction_excluding_carrier_line": energy_fracs,
    "spectral_energy_fraction_including_carrier_line": {f"{round(f/1e9):.0f}GHz": frac_in_with_dc(f) for f in (20e9, 27e9, TX_F3DB)},
    "carrier_line_share_of_total_energy": carrier_share,
    "B2_beta2_L_quarter_reach_km_1310": L_b2["1310 nm"] / 1e3, "B2_beta2_L_quarter_reach_km_1550": L_b2["1550 nm"] / 1e3,
}
results["experiments_to_try"] = {
    "exp1_eye_opening_10km_chirp_free_1310": op_10km["1310 nm"], "exp1_eye_opening_10km_chirp_free_1550": op_10km["1550 nm"],
    "exp1_eye_opening_1310_chirp_free_vs_km": {f"{Lz/1e3:.0f}": v for Lz, v in op_1310_far.items()},
    "exp2_T0_ps": T0_short * 1e12, "exp2_fwhm_in_ps": short["1310 nm"]["fwhm_in"] * 1e12,
    **{f"exp2_L_D_m_{n[:4]}": s["LD"] for n, s in short.items()},
    **{f"exp2_fwhm_out_2km_ps_{n[:4]}": s["fwhm_out"] * 1e12 for n, s in short.items()},
    "exp3_alpha": -ALPHA_CHIRP, "exp3_eye_opening_1550_vs_km": {f"{Lz/1e3:.1f}": v for Lz, v in op_neg.items()},
    "exp4_tx_rx_f3dB_GHz": F_WIDE / 1e9, "exp4_eye_height_b2b": float(h_b2b_w),
    "exp4_rms_optical_bandwidth_GHz": float(rms_bw(E_tx_w) / 1e9),
    **{f"exp4_eye_opening_2km_{n[:4]}": float(w["opening"]) for n, w in wide.items()},
    **{f"exp4_delay_spread_2km_ps_{n[:4]}": float(w["spread"] * 1e12) for n, w in wide.items()},
    "exp5_zdw_nm": zdw * 1e9, "exp5_beta2_ps2_km": silica.beta2(w_zdw) * 1e27, "exp5_beta3_ps3_km": silica.beta3(w_zdw) * 1e39,
    "exp5_gauss_fwhm_2km_ps": fwhm(t_g, P_zdw) * 1e12, "exp5_gauss_fwhm_10km_ps": fwhm(t_g, P_zdw_10) * 1e12,
    "exp5_eye_opening_10km_chirp_free": op_zdw_10km,
}

# figure: waveforms
seg = slice(0, 24 * SPB)
fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axs[0].step(t_s[seg] * 1e12, drive[seg], where="post", color=PALETTE["muted"], lw=1, label="NRZ drive (ideal)")
axs[0].plot(t_s[seg] * 1e12, P_tx[seg], color=PALETTE["ink"], label=f"launched optical power (Tx {TX_F3DB/1e9:.0f} GHz)")
axs[0].set_ylabel("power [a.u.]")
axs[0].set_title("The same 24 bits of the PRBS at the input and after 2 km of silica at 1310 nm and 1550 nm")
for ax, (name, d) in zip(axs[1:], nrz.items()):
    ax.plot(t_s[seg] * 1e12, P_tx[seg], color=PALETTE["muted"], lw=1, label="launched")
    ax.plot(t_s[seg] * 1e12, np.abs(d["E_out"][seg]) ** 2, color=COL[name], label=f"received optical power, {name}, z = 2 km")
    # advance the filtered trace by the filter's own group delay (the waveform is periodic, so a roll is exact)
    y_aligned = np.roll(d["y"], -int(round(TAU_RX / dt_s)))
    ax.plot(t_s[seg] * 1e12, y_aligned[seg], color=PALETTE["ink"], lw=1, ls="--",
            label=f"after Rx filter (its {TAU_RX*1e12:.1f} ps group delay removed)")
    ax.set_ylabel("power [a.u.]")
for ax in axs:
    # the traces live between 0 and ~1.1; leave a clear band above them and put the legend there in one row
    ax.set_ylim(-0.1, 1.75); ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)
axs[-1].set_xlabel("retarded time t − β₁L [ps]")
fig.tight_layout(); fig.savefig(OUT / "nrz_waveforms.png"); plt.close(fig)

# figure: spectrum (top) and the relative group delay of each component (bottom), sharing the frequency axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
S = S_full / S_full.max()
order = np.argsort(f_s)
ax1.semilogy(f_s[order] / 1e9, S[order], color=PALETTE["ink2"], lw=1, label="optical power spectrum of the NRZ field")
ax1.set_ylim(1e-7, 2); ax1.set_xlim(-150, 150)
ax1.set_ylabel("|Ã(Ω)|² (normalised)")
ax1.set_title(f"Top: the NRZ field spectrum; {100*energy_fracs['20GHz']:.0f} % of the modulation energy (carrier line excluded)\n"
              f"sits within ±20 GHz, {100*energy_fracs[f'{round(TX_F3DB/1e9):.0f}GHz']:.0f} % within the ±{TX_F3DB/1e9:.0f} GHz transmitter band (shaded)", fontsize=9)
fq = np.linspace(-150e9, 150e9, 400)
for name, m in mat.items():
    tau = m["beta2"] * L_LINK * 2 * np.pi * fq
    ax2.plot(fq / 1e9, tau * 1e12, color=COL[name], label=f"relative group delay β₂LΩ, {name}")
ax2.axhline(T_BIT * 1e12, color=PALETTE["red"], ls=":", lw=1, label="± one bit period T_b")
ax2.axhline(-T_BIT * 1e12, color=PALETTE["red"], ls=":", lw=1)
ax2.set_ylabel("delay relative to the carrier [ps]"); ax2.set_xlabel("frequency offset from carrier Ω/2π [GHz]")
ax2.set_title(f"Bottom: each component's delay after {L_LINK/1e3:.0f} km; at 1550 nm it spans ±{nrz['1550 nm']['spread']/2*1e12:.0f} ps across the band,\n"
              f"more than a bit period; at 1310 nm only ±{nrz['1310 nm']['spread']/2*1e12:.1f} ps", fontsize=9)
for ax in (ax1, ax2):
    ax.axvspan(-TX_F3DB / 1e9, TX_F3DB / 1e9, color=SERIES[2], alpha=0.08, label="±f_3dB transmitter band" if ax is ax1 else None)
    ax.legend(loc="upper left" if ax is ax1 else "upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "nrz_spectrum_delay.png"); plt.close(fig)


# figure: eye diagrams
def draw_eye(ax, y, color, title):
    tr = eye_fold(y)
    tt = np.arange(2 * SPB) * dt_s * 1e12
    segs = [np.column_stack([tt, r]) for r in tr]
    ax.add_collection(LineCollection(segs, colors=color, alpha=0.12, linewidths=0.8))
    ax.set_xlim(0, 2 * T_BIT * 1e12); ax.set_ylim(-0.1, 1.25)
    ax.set_xlabel("time [ps]  (two unit intervals)"); ax.set_ylabel("received power [a.u.]"); ax.set_title(title)


fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
draw_eye(axs[0], y_b2b, PALETTE["ink2"], f"Back-to-back (z = 0): eye height {m_b2b['height']:.2f}")
for ax, (name, d) in zip(axs[1:], nrz.items()):
    draw_eye(ax, d["y"], COL[name], f"{name}, 2 km: eye {100*d['opening']:.0f} % of back-to-back")
fig.suptitle(f"Chirp-free Tx after 2 km of silica: the 1310 nm eye is untouched, the 1550 nm eye has lost "
             f"{100*(1-nrz['1550 nm']['opening']):.0f} % of its height", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "eye_diagrams.png", bbox_inches="tight"); plt.close(fig)

# figure: eye diagrams with the chirped (DML-like) transmitter
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
draw_eye(axs[0], receive(E_tx_chirp), PALETTE["ink2"], f"Back-to-back, α = {ALPHA_CHIRP:.0f}: same eye (chirp is phase only)")
for ax, (name, d) in zip(axs[1:], nrz_chirp.items()):
    draw_eye(ax, d["y"], COL[name], f"{name}, 2 km, α = {ALPHA_CHIRP:.0f}: eye {100*max(d['opening'], 0):.0f} % of back-to-back")
fig.suptitle(f"A chirped transmitter (α = {ALPHA_CHIRP:.0f}, blue-shifted rising edges) widens the spectrum: "
             f"at 1550 nm the eye is closed after 2 km, at 1310 nm it survives", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "eye_diagrams_chirped.png", bbox_inches="tight"); plt.close(fig)

# figure: eye opening vs length
fig, ax = plt.subplots(figsize=(8, 4.2))
for name in LAMS:
    ax.plot(L_sweep / 1e3, 100 * sweep[name]["op"], color=COL[name],
            label=f"{name}, chirp-free (D = {mat[name]['D_ps_nm_km']:.1f} ps/(nm·km))")
    ax.plot(L_sweep / 1e3, 100 * np.clip(sweep[name + " chirped"]["op"], 0, None), color=COL[name], ls="--",
            label=f"{name}, chirped α = {ALPHA_CHIRP:.0f}")
ax.axhline(50, color=PALETTE["muted"], ls=":", lw=1); ax.axvline(2, color=PALETTE["muted"], ls="--", lw=1)
ax.text(2.05, 95, "2 km link", fontsize=8, color=PALETTE["ink2"])
ax.set_xlabel("link length L [km]"); ax.set_ylabel("eye opening [% of back-to-back]"); ax.set_ylim(-5, 115)
ax.set_title("Reach to a half-closed eye at %.3f Gbaud (chirp-free / chirped):\n1310 nm %s / %s,  1550 nm %s / %s"
             % (BAUD / 1e9, reach_str("1310 nm"), reach_str("1310 nm chirped"), reach_str("1550 nm"), reach_str("1550 nm chirped")), fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "eye_vs_length.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 5. videos
log("\n== 5. Videos ==")
FPS = 30
# 5a Gaussian envelope along z (retarded frame), both wavelengths, with the chirp underneath
N_FR = 240
z_frames = np.linspace(0, L_LINK, N_FR)
frames_P = {name: [] for name in LAMS}; frames_df = {name: [] for name in LAMS}
for name, m in mat.items():
    for z in z_frames:
        A = propagate_retarded(A0, dt_g, m["omega"], z)
        frames_P[name].append(np.abs(A) ** 2); frames_df[name].append(inst_freq_shift(A, dt_g))
tmask = (t_g > -40e-12) & (t_g < 40e-12)


def gauss_fig():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    lines = {}
    a1.plot(t_g[tmask] * 1e12, P_in[tmask], color=PALETTE["muted"], ls="--", lw=1, label="input")
    for name in LAMS:
        lines[name] = a1.plot([], [], color=COL[name], label=f"{name}")[0]
        lines[name + "c"] = a2.plot([], [], color=COL[name])[0]
    a1.set_ylim(0, 1.05); a1.set_ylabel("|A(z,t)|²"); a1.legend(loc="upper right")
    a2.set_ylim(-150, 150); a2.set_ylabel("inst. frequency − f_c [GHz]"); a2.set_xlabel("retarded time t − β₁z [ps]")
    a2.axhline(0, color=PALETTE["muted"], lw=0.8)
    a2.set_title("The pulse acquires a chirp: blue leads, red lags (β₂ < 0)")
    ttl = a1.set_title("")
    return fig, lines, ttl


def gauss_update(i, lines, ttl):
    for name in LAMS:
        P = frames_P[name][i]
        lines[name].set_data(t_g[tmask] * 1e12, P[tmask])
        df = frames_df[name][i]; vis = tmask & (P > 0.02 * P.max())
        lines[name + "c"].set_data(t_g[vis] * 1e12, df[vis] / 1e9)
    ttl.set_text(f"Gaussian pulse (T₀ = {T0_GAUSS*1e12:.0f} ps) in bulk silica, z = {z_frames[i]/1e3:.2f} km: "
                 f"1550 nm spreads, 1310 nm keeps its shape")


fig, lines, ttl = gauss_fig()
anim = FuncAnimation(fig, gauss_update, frames=N_FR, fargs=(lines, ttl), blit=False)
anim.save(OUT / "gaussian_envelope.mp4", writer=FFMpegWriter(fps=FPS, bitrate=1800)); plt.close(fig)
# contact sheet
sel = [0, 60, 120, 180, 239]
fig, axs = plt.subplots(1, len(sel), figsize=(16, 3.2))
for ax, i in zip(axs, sel):
    for name in LAMS:
        ax.plot(t_g[tmask] * 1e12, frames_P[name][i][tmask], color=COL[name], label=name)
    ax.set_ylim(0, 1.05); ax.set_title(f"z = {z_frames[i]/1e3:.2f} km", fontsize=9); ax.set_xlabel("t − β₁z [ps]")
axs[0].set_ylabel("|A|²"); axs[0].legend(fontsize=8)
fig.suptitle("gaussian_envelope.mp4 stills: the 1550 nm pulse spreads along z, the 1310 nm pulse does not", y=1.03)
fig.tight_layout(); fig.savefig(OUT / "gaussian_envelope_frames.png", bbox_inches="tight"); plt.close(fig)
log("gaussian_envelope.mp4 written (%d frames)" % N_FR)

# 5b NRZ eye closing along z: left waveform, right eye, rows = wavelengths
N_FR2 = 200
z2 = np.linspace(0, L_LINK, N_FR2)
eye_frames = {name: [] for name in LAMS}
for name, m in mat.items():
    for z in z2:
        eye_frames[name].append(receive(propagate_retarded(E_tx, dt_s, m["omega"], z)))
seg2 = slice(0, 16 * SPB)
tt2 = np.arange(2 * SPB) * dt_s * 1e12


def eye_fig():
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    art = {}
    for r, name in enumerate(LAMS):
        axw, axe = axs[r]
        axw.plot(t_s[seg2] * 1e12, y_b2b[seg2], color=PALETTE["muted"], lw=1, label="back-to-back")
        art[name + "w"] = axw.plot([], [], color=COL[name], label=f"received, {name}")[0]
        axw.set_xlim(0, t_s[seg2][-1] * 1e12); axw.set_ylim(-0.1, 1.25); axw.set_ylabel("received power [a.u.]")
        axw.legend(loc="upper right", fontsize=8)
        lc = LineCollection([], colors=COL[name], alpha=0.12, linewidths=0.8); axe.add_collection(lc)
        art[name + "e"] = lc
        axe.set_xlim(0, 2 * T_BIT * 1e12); axe.set_ylim(-0.1, 1.25); axe.set_ylabel("power [a.u.]")
        art[name + "t"] = axe.set_title("")
    axs[1, 0].set_xlabel("retarded time [ps]"); axs[1, 1].set_xlabel("time [ps] (two unit intervals)")
    art["sup"] = fig.suptitle("")
    fig.tight_layout()
    return fig, art


def eye_update(i, art):
    for name in LAMS:
        y = eye_frames[name][i]
        art[name + "w"].set_data(t_s[seg2] * 1e12, y[seg2])
        art[name + "e"].set_segments([np.column_stack([tt2, r]) for r in eye_fold(y)])
        h = eye_metrics(y)["height"] / m_b2b["height"]
        art[name + "t"].set_text(f"{name}: eye opening {100*max(h,0):.0f} %")
    art["sup"].set_text(f"53.125 Gbaud NRZ after z = {z2[i]/1e3:.2f} km of silica (material dispersion only)")


fig, art = eye_fig()
anim = FuncAnimation(fig, eye_update, frames=N_FR2, fargs=(art,), blit=False)
anim.save(OUT / "nrz_eye_closing.mp4", writer=FFMpegWriter(fps=FPS, bitrate=2200)); plt.close(fig)
sel2 = [0, 50, 100, 150, 199]
fig, axs = plt.subplots(2, len(sel2), figsize=(16, 5.6))
for c, i in enumerate(sel2):
    for r, name in enumerate(LAMS):
        draw_eye(axs[r, c], eye_frames[name][i], COL[name], f"{name}, z = {z2[i]/1e3:.2f} km")
        axs[r, c].set_title(f"{name}, z = {z2[i]/1e3:.2f} km", fontsize=9)
        if c: axs[r, c].set_ylabel("")
        if r == 0: axs[r, c].set_xlabel("")
fig.suptitle("nrz_eye_closing.mp4 stills (chirp-free Tx): the 1550 nm eye shrinks along the 2 km, the 1310 nm eye stays open", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "nrz_eye_closing_frames.png", bbox_inches="tight"); plt.close(fig)
log("nrz_eye_closing.mp4 written (%d frames)" % N_FR2)


# ------------------------------------------------------------------ 6. capstone numbers
log("\n== 6. Capstone connection numbers ==")
tau_ph = REF.q_loaded * REF.lambda_nm * 1e-9 / (2 * np.pi * C0)      # photon lifetime = Q/omega
fwhm_ghz = C0 * REF.fwhm_pm * 1e-12 / (REF.lambda_nm * 1e-9) ** 2 / 1e9
dlam_signal_1310 = nrz["1310 nm"]["dlam"] * 1e12                        # pm
log(f"ring photon lifetime Q/omega = {tau_ph*1e12:.2f} ps; ring FWHM {REF.fwhm_pm} pm = {fwhm_ghz:.1f} GHz")
log(f"NRZ optical bandwidth at 1310 nm: {dlam_signal_1310:.0f} pm double-sided vs ring FWHM {REF.fwhm_pm} pm and thermal drift {REF.dlambda_dT_pm_per_K} pm/K")
log(f"the whole 2 km of silica at 1310 gives a delay spread of {nrz['1310 nm']['spread']*1e12:.1f} ps; the ring's own lifetime {tau_ph*1e12:.1f} ps is the same order")
# the 8-channel, 200 GHz O-band grid centred on 1310 nm: D at every channel and the worst-channel delay spread over 2 km
f_c = C0 / (REF.lambda_nm * 1e-9)
f_grid = f_c + (np.arange(REF.n_channels) - (REF.n_channels - 1) / 2) * REF.channel_spacing_ghz * 1e9
lam_grid = C0 / f_grid
D_grid = silica.D_ps_nm_km(lam_grid)
beta2_grid = np.array([silica.beta2(2 * np.pi * f) for f in f_grid])
spread_grid = np.abs(beta2_grid) * L_LINK * 2 * np.pi * 2 * TX_F3DB          # same definition as the NRZ delay spread above
skew_grid = (silica.beta1(2 * np.pi * f_grid[0]) - silica.beta1(2 * np.pi * f_grid[-1])) * L_LINK
log(f"8-channel grid, {REF.channel_spacing_ghz:.0f} GHz spacing: lambda {lam_grid.min()*1e9:.1f}..{lam_grid.max()*1e9:.1f} nm, "
    f"D {D_grid.min():.2f}..{D_grid.max():.2f} ps/(nm km), worst-channel delay spread over 2 km {spread_grid.max()*1e12:.1f} ps "
    f"({100*spread_grid.max()/T_BIT:.0f} % of a bit), channel-to-channel arrival skew {abs(skew_grid)*1e12:.0f} ps")
results["capstone"] = {
    "ring_photon_lifetime_ps": tau_ph * 1e12, "ring_fwhm_GHz": fwhm_ghz,
    "nrz_optical_bandwidth_pm_1310": dlam_signal_1310, "thermal_drift_pm_per_K": REF.dlambda_dT_pm_per_K,
    "delay_spread_2km_1310_ps": nrz["1310 nm"]["spread"] * 1e12, "bit_period_ps": T_BIT * 1e12,
    "delay_spread_2km_1310_fraction_of_bit": nrz["1310 nm"]["spread"] / T_BIT,
    "delay_spread_2km_1550_fraction_of_bit": nrz["1550 nm"]["spread"] / T_BIT,
    "wdm_grid_lambda_nm": [float(x) for x in lam_grid * 1e9],
    "wdm_grid_D_ps_nm_km": [float(x) for x in D_grid],
    "wdm_grid_worst_delay_spread_2km_ps": float(spread_grid.max() * 1e12),
    "wdm_grid_edge_to_edge_skew_2km_ps": float(abs(skew_grid) * 1e12),
}

# ------------------------------------------------------------------ write outputs
results["runtime_s"] = time.time() - T_START
(OUT / "results.json").write_text(json.dumps(results, indent=2))
(OUT / "results.txt").write_text("\n".join(log_lines) + "\n")
tools = [
    {"tool": "numpy (FFT)", "version": np.__version__,
     "what_it_is": "The array library of scientific Python; numpy.fft is its fast Fourier transform, used everywhere a signal is moved between time and frequency.",
     "used_for": "The split-into-frequencies propagation of notes section 12: FFT the input field, multiply each frequency component by exp(-j beta(omega) L) with the exact Sellmeier beta(omega), inverse FFT. Also the NRZ transmitter/receiver filtering (all in the frequency domain) and the eye-diagram folding.",
     "result": f"Gaussian FWHM after 2 km: 1550 nm {gauss['1550 nm']['fwhm_out']*1e12:.2f} ps vs analytic {gauss['1550 nm']['fwhm_an']*1e12:.2f} ps ({pct(gauss['1550 nm']['fwhm_out'], gauss['1550 nm']['fwhm_an']):.3f} %), 1310 nm {gauss['1310 nm']['fwhm_out']*1e12:.2f} ps vs {gauss['1310 nm']['fwhm_an']*1e12:.2f} ps; group delay at 1310 nm {delay['1310 nm']['t_meas']*1e6:.4f} us vs beta1 L = n_g L/c {delay['1310 nm']['t_g']*1e6:.4f} us ({results['group_delay']['t_g_agreement_pct_1310']:.4f} %); chirp-free NRZ eye opening after 2 km: 1310 nm {100*nrz['1310 nm']['opening']:.0f} %, 1550 nm {100*nrz['1550 nm']['opening']:.0f} % of back-to-back; with a chirped Tx (alpha = {ALPHA_CHIRP:.0f}): 1310 nm {100*nrz_chirp['1310 nm']['opening']:.0f} %, 1550 nm {100*max(nrz_chirp['1550 nm']['opening'],0):.0f} % (closed).",
     "how_to_observe": "cd experiments/05_pulse_dispersion && ../../.venv/bin/python run.py; look at out/gaussian_broadening.png, out/group_delay.png, out/eye_diagrams.png, out/eye_diagrams_chirped.png, out/eye_vs_length.png. Change L_LINK, T0_GAUSS, TX_F3DB, ALPHA_CHIRP or BAUD at the top of run.py and re-run."},
    {"tool": "sympy", "version": sympy.__version__,
     "what_it_is": "A symbolic mathematics library: exact algebra, differentiation and integration on expressions, with lambdify to turn the result into fast numpy functions.",
     "used_for": "Differentiating the Sellmeier n(lambda) exactly to get n_g = n - lambda dn/dlambda, D = -(lambda/c) d2n/dlambda2 and beta1, beta2, beta3 = d^k beta/d omega^k of beta(omega) = n(omega) omega/c, so no finite-difference error enters the propagator (silica.py).",
     "result": f"n(1310) = {mat['1310 nm']['n']:.4f} (notes: 1.4468), n_g(1310) = {mat['1310 nm']['ng']:.4f}, D(1310) = {mat['1310 nm']['D_ps_nm_km']:.2f}, D(1550) = {mat['1550 nm']['D_ps_nm_km']:.2f} ps/(nm km) (bulk expectation 20-22), beta2(1550) = {mat['1550 nm']['beta2']*1e27:.2f} ps^2/km equal to -lambda^2 D/(2 pi c) to {results['material']['beta2_vs_formula_agreement_pct']:.6f} %; zero-dispersion wavelength {zdw*1e9:.0f} nm.",
     "how_to_observe": "out/silica_dispersion.png and the first block of out/results.txt; `../../.venv/bin/python -c \"import silica; print(silica.symbolic_summary())\"`. Edit B_SELL / C_SELL_UM2 in silica.py to try another glass."},
    {"tool": "scipy (signal + optimize)", "version": scipy.__version__,
     "what_it_is": "SciPy's signal-processing module: filter design (Bessel, Butterworth), frequency responses, and test sequences such as maximal-length PRBS; scipy.optimize holds the root finders.",
     "used_for": "signal.max_len_seq(9) for the 511-bit PRBS, signal.bessel(4, ...) + signal.freqs() for the receiver's 4th-order Bessel low-pass (0.75 x baud) evaluated on the FFT grid (the same response gives the filter's DC group delay, which the waveform figure removes so it is not mistaken for a dispersion delay); optimize.brentq for the zero-dispersion wavelength of the Sellmeier fit and for the exact length at which the eye opening crosses 50 % (bracketed by the 250 m sweep, solved to 1 m).",
     "result": f"Back-to-back eye height {m_b2b['height']:.3f} (of P1 = 1); Rx Bessel DC group delay {TAU_RX*1e12:.2f} ps = {TAU_RX/T_BIT:.2f} UI; length at which the eye is half closed (brentq, 1 m tolerance): 1310 nm {reach_str('1310 nm')} chirp-free / {reach_str('1310 nm chirped')} chirped, 1550 nm {reach_str('1550 nm')} chirp-free / {reach_str('1550 nm chirped')} chirped; zero-dispersion wavelength {zdw*1e9:.1f} nm (bulk silica, expected ~1270 nm).",
     "how_to_observe": "out/nrz_waveforms.png, out/nrz_spectrum_delay.png, out/eye_diagrams.png, out/eye_vs_length.png. Change RX_F3DB, EXT_RATIO_DB or PRBS_ORDER in run.py and re-run."},
    {"tool": f"matplotlib (+ FFMpegWriter / ffmpeg {ffmpeg_version()})", "version": matplotlib.__version__,
     "what_it_is": "The standard Python plotting library; its animation module drives ffmpeg to encode a sequence of redrawn frames into an mp4.",
     "used_for": "All figures, the eye diagrams (LineCollection of folded traces), and two videos: the Gaussian envelope and its chirp evolving along z, and the NRZ waveform plus eye diagram evolving along z.",
     "result": f"out/gaussian_envelope.mp4 ({N_FR/FPS:.0f} s, {N_FR} frames) and out/nrz_eye_closing.mp4 ({N_FR2/FPS:.1f} s, {N_FR2} frames) with contact sheets out/gaussian_envelope_frames.png and out/nrz_eye_closing_frames.png; {len(list(OUT.glob('*.png')))} PNG figures in total.",
     "how_to_observe": "open out/gaussian_envelope.mp4: watch the orange (1550 nm) pulse widen while the blue (1310 nm) pulse keeps its shape, and the lower panel show the linear chirp growing; open out/nrz_eye_closing.mp4: the right column eye at 1550 nm shrinks along the 2 km while the 1310 nm eye does not move. N_FR / N_FR2 set the frame counts, FPS the frame rate."},
]
(OUT / "tools.json").write_text(json.dumps(tools, indent=2))
log(f"\nDone in {results['runtime_s']:.1f} s. Outputs in {OUT}")
