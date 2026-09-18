"""03_driven_electron_rlc: the Lorentz oscillator as a series RLC circuit (ngspice + scipy + notebook).

Notes sections 6, 7 and 8.  The bound electron  m x'' + m*gamma x' + m*w0^2 x = -q E(t)  is the same ODE as
the series RLC  L q'' + R q' + q/C = V(t).  We let ngspice solve the circuit (AC sweep and transients at
three drive frequencies), overlay the analytic chi(w) = A/(w0^2 - w^2 + j*gamma*w), show the power argument
(in phase = no absorption, 90 deg = maximum absorption) with the SPICE resistor power, map the dimensionless
result onto the UV resonance of fused silica, compute n' and n'', and finish with the capstone link: free
carriers (Drude = Lorentz with w0 -> 0) and the thermo-optic effect in the same language.

Run:  cd experiments/03_driven_electron_rlc && ../../.venv/bin/python run.py
Everything in out/ is deleted and regenerated.  explore.ipynb is executed (not rebuilt) at the end.
"""
import sys, pathlib, json, subprocess, time, textwrap
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE, C0, Q_E, M_E, EPS0
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lorentz import (chi_hat, phase_lag_deg, rlc_transfer, rlc_absorbed_power, rlc_values,
                     n_complex_from_chi, lorentz_sum, malitson_oscillators, sellmeier_n,
                     effective_uv_oscillator, drude_chi_hat, drude_silicon, soref_bennett_1310,
                     sellmeier_dn_dlam_terms)

NGSPICE = "/opt/homebrew/bin/ngspice"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
JUPYTER = ROOT / ".venv" / "bin" / "jupyter"
T_START = time.time()
RESULTS = {}
LOG = []

# regenerate from scratch: remove every previous output
for f in OUT.iterdir():
    if f.is_file():
        f.unlink()


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def save_results(stage):
    """Write results.json and run_log.txt now, so an interrupted run (the video and the notebook take most of
    the time) never leaves out/ without the headline-number file.  Called after every stage and at the end."""
    RESULTS["stage_completed"] = stage
    RESULTS["runtime_s"] = time.time() - T_START
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))
    (OUT / "run_log.txt").write_text("\n".join(LOG))


def wrap_title(ax, text, width=62, size=9):
    ax.set_title("\n".join(textwrap.wrap(text, width)), fontsize=size)


def fmt_power(P_W):
    """Power with a unit that keeps three significant figures: 0.839 µW, 11.2 µW, 7.95 mW."""
    return f"{P_W*1e6:.3g} µW" if abs(P_W) < 1e-4 else f"{P_W*1e3:.3g} mW"


def phasor_axes(ax, xlabel="Re", ylabel="Im"):
    """Unit-circle phasor panel with room below the circle for the legend.

    Arrows are unit length (directions only).  The y range is extended down to -2.6 so a lower-left legend sits
    entirely below the circle and can never cross an arrow (the arrow pointing to (0, -1) at resonance is the
    point of the figure); ticks are limited to the circle's range."""
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-2.6, 1.3)
    ax.set_yticks([-1, 0, 1]); ax.set_xticks([-1, 0, 1])
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=PALETTE["line"], lw=0.8, zorder=0)
    ax.axhline(0, color=PALETTE["line"], lw=0.6, zorder=0); ax.axvline(0, color=PALETTE["line"], lw=0.6, zorder=0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


# ----------------------------------------------------------------------------------------------
# 1. The circuit: f0 = 1 kHz, Q = 10 (Gamma = gamma/w0 = 0.1), L = 100 mH
# ----------------------------------------------------------------------------------------------
F0 = 1000.0
Q_FACTOR = 10.0
L, R, C, W0, GAMMA = rlc_values(F0, Q_FACTOR, L=0.1)
V0 = 1.0                                   # drive amplitude, volts (plays the role of -q_e E0)
GAMMA_HAT = GAMMA / W0
log(f"Series RLC: L = {L*1e3:.1f} mH, R = {R:.3f} ohm, C = {C*1e9:.2f} nF")
log(f"  -> w0 = 1/sqrt(LC) = {W0:.2f} rad/s  (f0 = {W0/2/np.pi:.2f} Hz), gamma = R/L = {GAMMA:.2f} 1/s, "
    f"Gamma = gamma/w0 = {GAMMA_HAT:.3f} (Q = {1/GAMMA_HAT:.1f})")
RESULTS["circuit"] = {"L_H": L, "R_ohm": R, "C_F": C, "f0_Hz": F0, "w0_rad_s": W0, "gamma_1_s": GAMMA,
                      "Gamma_dimensionless": GAMMA_HAT, "Q": Q_FACTOR, "V0_V": V0}

# ----------------------------------------------------------------------------------------------
# 2. Netlists.  The capacitor charge q_C = C * v(cap) is the "displacement"; i = dq/dt the "velocity".
#    ngspice reports i(v1) flowing INTO the source's + terminal, so the loop current is -i(v1).
# ----------------------------------------------------------------------------------------------
def netlist_header(title, L=L, R=R, C=C):
    return (f"* {title}\n"
            f"* Lorentz map: m<->L, m*gamma<->R, m*w0^2<->1/C, -qE<->V, x<->q_C=C*v(cap)\n"
            f"V1 in 0 dc 0 ac {V0} sin(0 {V0} {{FDRIVE}})\n"
            f"L1 in n1 {L:.6g}\n"
            f"R1 n1 cap {R:.6g}\n"
            f"C1 cap 0 {C:.6g}\n")


AC_PTS_PER_DEC = 1000


def write_ac_netlist(cir, txt, title, L=L, R=R, C=C):
    cir.write_text(
        netlist_header(title, L, R, C).replace("{FDRIVE}", "1000")
        + ".control\n"
          "set filetype=ascii\nset wr_vecnames\nset wr_singlescale\n"
          f"ac dec {AC_PTS_PER_DEC} 10 100k\n"
          "let i_loop = -i(v1)\n"
          f"wrdata {txt} v(cap) i_loop\n"
          "quit\n.endc\n.end\n")


AC_TXT = OUT / "rlc_ac.txt"
ac_cir = OUT / "rlc_ac.cir"
write_ac_netlist(ac_cir, AC_TXT, "series RLC: AC sweep of capacitor voltage (charge) and loop current")

DRIVES = {"low": 0.1 * F0, "res": 1.0 * F0, "high": 3.0 * F0}      # Omega = 0.1, 1, 3
SETTLE_OVER_GAMMA = 16.0                    # ring-down of the start-up transient is e^{-gamma t/2}: 16/gamma -> e^-8
TRAN_FILES = {}
for key, fd in DRIVES.items():
    period = 1.0 / fd
    t_settle = SETTLE_OVER_GAMMA / GAMMA
    t_stop = t_settle + 4 * period
    t_step = period / 400
    txt = OUT / f"rlc_tran_{key}.txt"
    cir = OUT / f"rlc_tran_{key}.cir"
    cir.write_text(
        netlist_header(f"series RLC: transient, drive {fd:g} Hz (Omega = {fd/F0:g})").replace("{FDRIVE}", f"{fd:g}")
        + ".control\n"
          "set filetype=ascii\nset wr_vecnames\nset wr_singlescale\n"
          f"tran {t_step:.4g} {t_stop:.4g} 0 {t_step:.4g}\n"
          "let i_loop = -i(v1)\n"
          "let p_r = (v(n1)-v(cap))*i_loop\n"      # instantaneous resistor power  (R i^2)
          "let p_src = v(in)*i_loop\n"             # instantaneous power delivered by the source (E * velocity)
          f"wrdata {txt} v(in) v(cap) i_loop p_r p_src\n"
          "quit\n.endc\n.end\n")
    TRAN_FILES[key] = (cir, txt, fd, t_settle, t_stop)


def run_ngspice(cir):
    t0 = time.time()
    p = subprocess.run([NGSPICE, "-b", str(cir)], capture_output=True, text=True, cwd=OUT)
    (OUT / (cir.stem + ".log")).write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"ngspice failed on {cir}:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    log(f"  ngspice {cir.name}: {time.time()-t0:.2f} s")


def read_wrdata(txt):
    """wrdata ASCII table: header line then whitespace-separated numbers. Complex vectors take two columns."""
    with open(txt) as f:
        header = f.readline().split()
    data = np.loadtxt(txt, skiprows=1)
    return header, data


log("Running ngspice ...")
run_ngspice(ac_cir)
for key in TRAN_FILES:
    run_ngspice(TRAN_FILES[key][0])

# ----------------------------------------------------------------------------------------------
# 3. AC sweep: SPICE H(w) = q_C / V versus analytic 1/(L (w0^2 - w^2 + j gamma w))
# ----------------------------------------------------------------------------------------------
hdr, ac = read_wrdata(AC_TXT)
f_ac = ac[:, 0]
vcap = ac[:, 1] + 1j * ac[:, 2]           # complex capacitor voltage phasor for V = 1 V
i_ac = ac[:, 3] + 1j * ac[:, 4]           # loop current phasor
H_spice = C * vcap / V0                   # charge per volt  [C/V]
H_an = rlc_transfer(f_ac, L, R, C)
Om = f_ac / F0
chi_spice = W0**2 * L * H_spice           # dimensionless  chi_hat = chi w0^2 / A
chi_an = chi_hat(Om, GAMMA_HAT)
P_spice_ac = 0.5 * np.real(V0 * np.conj(i_ac))         # <P> = 1/2 Re(V I*)
P_an_ac = rlc_absorbed_power(f_ac, L, R, C, V0)

err_mag = np.max(np.abs(np.abs(chi_spice) - np.abs(chi_an)) / np.abs(chi_an))
err_ph = np.max(np.abs(np.degrees(np.angle(chi_spice)) - np.degrees(np.angle(chi_an))))
log(f"AC sweep ({len(f_ac)} points): max |chi| relative error SPICE vs analytic = {err_mag*100:.5f} %, "
    f"max phase error = {err_ph:.5f} deg")


def parabolic_peak(x, y):
    """Vertex of the parabola through the three samples around the maximum of y (x assumed log-spaced -> use log x)."""
    i = int(np.argmax(y)); i = min(max(i, 1), len(y) - 2)
    lx = np.log10(x[i - 1:i + 2]); yy = y[i - 1:i + 2]
    a, b, _ = np.polyfit(lx, yy, 2)
    return 10 ** (-b / (2 * a))


def half_power_width(x, y):
    """Full width at half maximum with linear interpolation of the two crossings."""
    i = int(np.argmax(y)); half = y[i] / 2
    j = i
    while j > 0 and y[j] >= half: j -= 1
    x_lo = np.interp(half, [y[j], y[j + 1]], [x[j], x[j + 1]])
    k = i
    while k < len(y) - 1 and y[k] >= half: k += 1
    x_hi = np.interp(half, [y[k], y[k - 1]], [x[k], x[k - 1]])
    return x_hi - x_lo, x_lo, x_hi


# resonance of the displacement amplitude (peaks slightly BELOW w0 for finite damping)
f_pk_spice = parabolic_peak(f_ac, np.abs(H_spice))
f_pk_an = F0 * np.sqrt(1 - GAMMA_HAT**2 / 2)
ph_spice = -np.degrees(np.angle(H_spice))          # positive = lag
ph_f0_spice = np.interp(F0, f_ac, ph_spice)
# power FWHM: for the series RLC the half-power points satisfy |wL - 1/(wC)| = R, whose two positive roots
# differ by EXACTLY R/L = gamma (not only in the narrow-band limit)
fwhm_spice, f_lo, f_hi = half_power_width(f_ac, P_spice_ac)
fwhm_an = GAMMA / (2 * np.pi)
f_lo_an = (W0 * (-GAMMA_HAT / 2 + np.sqrt(1 + GAMMA_HAT**2 / 4))) / (2 * np.pi)
f_hi_an = (W0 * (+GAMMA_HAT / 2 + np.sqrt(1 + GAMMA_HAT**2 / 4))) / (2 * np.pi)
log(f"  |q| peak: SPICE {f_pk_spice:.2f} Hz, analytic f0*sqrt(1-Gamma^2/2) = {f_pk_an:.2f} Hz "
    f"({(f_pk_spice/f_pk_an-1)*100:+.3f} %)")
log(f"  phase lag at f0: SPICE {ph_f0_spice:.3f} deg (expect 90)")
log(f"  absorbed-power half-power points: SPICE {f_lo:.2f} / {f_hi:.2f} Hz, analytic {f_lo_an:.2f} / {f_hi_an:.2f} Hz")
log(f"  absorbed-power FWHM: SPICE {fwhm_spice:.3f} Hz, analytic gamma/2pi = {fwhm_an:.3f} Hz "
    f"({(fwhm_spice/fwhm_an-1)*100:+.3f} %)")
log(f"  static response |chi_hat|: SPICE {np.abs(chi_spice[0]):.5f} at {f_ac[0]:g} Hz, analytic {np.abs(chi_an[0]):.5f} "
    f"(quasi-static limit chi -> A/w0^2, i.e. chi_hat -> 1)")
RESULTS["ac_sweep"] = {
    "points": int(len(f_ac)),
    "max_rel_error_magnitude_pct": float(err_mag * 100), "max_phase_error_deg": float(err_ph),
    "f_peak_displacement_spice_Hz": float(f_pk_spice), "f_peak_displacement_analytic_Hz": float(f_pk_an),
    "phase_lag_at_f0_spice_deg": float(ph_f0_spice), "phase_lag_at_f0_expected_deg": 90.0,
    "power_half_points_spice_Hz": [float(f_lo), float(f_hi)], "power_half_points_analytic_Hz": [f_lo_an, f_hi_an],
    "power_fwhm_spice_Hz": float(fwhm_spice), "power_fwhm_analytic_Hz": float(fwhm_an),
    "static_chi_hat_spice": float(np.abs(chi_spice[0])), "static_chi_hat_analytic": float(np.abs(chi_an[0])),
}

# 3b. Two more AC sweeps at Q = 2 and Q = 50 (same f0, same L): the peak |chi_hat| ~ Q check that explore.ipynb
#     overlays is produced here as well, so its numbers live in results.json.  Analytic peak of |chi_hat| is
#     1/(Gamma*sqrt(1 - Gamma^2/4)) at Omega = sqrt(1 - Gamma^2/2).
RESULTS["ac_sweep_other_Q"] = {}
for Q_other in (2.0, 50.0):
    L_o, R_o, C_o, W0_o, _ = rlc_values(F0, Q_other, L=0.1)
    G_o = 1.0 / Q_other
    cir_o = OUT / f"rlc_ac_Q{Q_other:g}.cir"; txt_o = OUT / f"rlc_ac_Q{Q_other:g}.txt"
    write_ac_netlist(cir_o, txt_o, f"series RLC: AC sweep at Q = {Q_other:g} (R = {R_o:.4g} ohm)", L_o, R_o, C_o)
    run_ngspice(cir_o)
    _, ac_o = read_wrdata(txt_o)
    chi_o = W0_o**2 * L_o * C_o * (ac_o[:, 1] + 1j * ac_o[:, 2]) / V0
    chi_o_an = chi_hat(ac_o[:, 0] / F0, G_o)
    err_o = float(np.max(np.abs(np.abs(chi_o) - np.abs(chi_o_an)) / np.abs(chi_o_an)) * 100)
    peak_o = float(np.abs(chi_o).max())
    peak_o_an = float(1.0 / (G_o * np.sqrt(1 - G_o**2 / 4)))
    log(f"  AC sweep at Q = {Q_other:g} (R = {R_o:.2f} ohm): SPICE peak |chi_hat| = {peak_o:.4f}, analytic "
        f"1/(Gamma sqrt(1-Gamma^2/4)) = {peak_o_an:.4f} ({(peak_o/peak_o_an-1)*100:+.4f} %); max |chi| error {err_o:.5f} %")
    RESULTS["ac_sweep_other_Q"][f"Q{Q_other:g}"] = {
        "Q": Q_other, "R_ohm": float(R_o), "points": int(len(ac_o)),
        "peak_chi_hat_spice": peak_o, "peak_chi_hat_analytic": peak_o_an,
        "peak_rel_error_pct": float((peak_o / peak_o_an - 1) * 100), "max_rel_error_magnitude_pct": err_o,
    }
peak_main = float(np.abs(chi_spice).max()); peak_main_an = float(1.0 / (GAMMA_HAT * np.sqrt(1 - GAMMA_HAT**2 / 4)))
RESULTS["ac_sweep"]["peak_chi_hat_spice"] = peak_main; RESULTS["ac_sweep"]["peak_chi_hat_analytic"] = peak_main_an
log(f"  AC sweep at Q = {Q_FACTOR:g}: SPICE peak |chi_hat| = {peak_main:.4f}, analytic {peak_main_an:.4f}")

# Figure 1: AC sweep, four panels
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
step = AC_PTS_PER_DEC // 15
ax = axs[0, 0]
ax.loglog(Om, np.abs(chi_an), color=SERIES[0], label="analytic 1/|1−Ω²+jΓΩ|")
ax.loglog(Om[::step], np.abs(chi_spice[::step]), "o", ms=4, color=SERIES[1], label="ngspice C·v(cap)·ω₀²L/V₀")
ax.axvline(1, color=PALETTE["muted"], lw=0.8, ls="--")
ax.set_xlabel("drive frequency Ω = ω/ω₀ (–)   [circuit: f = Ω × 1 kHz]")
ax.set_ylabel("|χ̂| = |q_C| ω₀² L / V₀  (–)")
wrap_title(ax, "Displacement per unit drive is flat (=1) at low Ω, peaks at Ω ≈ 1 (×Q = 10), falls as 1/Ω² above")
ax.legend()
ax = axs[0, 1]
ax.semilogx(Om, phase_lag_deg(Om, GAMMA_HAT), color=SERIES[0], label="analytic atan2(ΓΩ, 1−Ω²)")
ax.semilogx(Om[::step], ph_spice[::step], "o", ms=4, color=SERIES[1], label="ngspice")
ax.axhline(90, color=PALETTE["muted"], lw=0.8, ls="--"); ax.axvline(1, color=PALETTE["muted"], lw=0.8, ls="--")
ax.set_xlabel("Ω = ω/ω₀ (–)"); ax.set_ylabel("phase lag of q_C behind V  (deg)")
wrap_title(ax, "The response lags the drive: 0° well below, 90° exactly at, 180° well above resonance")
ax.set_yticks([0, 45, 90, 135, 180]); ax.legend()
ax = axs[1, 0]
ax.semilogx(Om, chi_an.real, color=SERIES[0], label="Re χ̂ (in phase → refractive index)")
ax.semilogx(Om, -chi_an.imag, color=SERIES[2], label="−Im χ̂ (quadrature → absorption)")
ax.semilogx(Om[::step], chi_spice.real[::step], "o", ms=4, color=SERIES[0], label="ngspice Re χ̂")
ax.semilogx(Om[::step], -chi_spice.imag[::step], "s", ms=4, color=SERIES[2], label="ngspice −Im χ̂")
ax.axhline(0, color=PALETTE["muted"], lw=0.8); ax.axvline(1, color=PALETTE["muted"], lw=0.8, ls="--")
ax.set_xlabel("Ω = ω/ω₀ (–)"); ax.set_ylabel("χ̂ components (–)")
wrap_title(ax, "Re χ̂ is dispersive (positive below, negative above resonance); −Im χ̂ is a line of width Γ")
ax.legend(loc="upper left")
ax = axs[1, 1]
ax.semilogx(Om, P_an_ac * 1e3, color=SERIES[0], label="analytic ½V₀²R/|Z|²")
ax.semilogx(Om[::step], P_spice_ac[::step] * 1e3, "o", ms=4, color=SERIES[1], label="ngspice ½Re(V I*)")
ax.axvline(1, color=PALETTE["muted"], lw=0.8, ls="--")
ax.axvspan(f_lo / F0, f_hi / F0, color=SERIES[3], alpha=0.12, label=f"half-power band, FWHM {fwhm_spice:.1f} Hz")
ax.set_xlabel("Ω = ω/ω₀ (–)"); ax.set_ylabel("time-averaged absorbed power  (mW)")
wrap_title(ax, f"Power absorbed in R peaks at Ω = 1 with full width Γ = {GAMMA_HAT:g} (γ/2π = {fwhm_an:.0f} Hz)")
ax.legend()
fig.suptitle(f"Series RLC (L={L*1e3:.0f} mH, R={R:.1f} Ω, C={C*1e9:.0f} nF, f₀=1 kHz, Q=10) ≡ Lorentz oscillator: "
             f"ngspice AC sweep vs χ(ω)=A/(ω₀²−ω²+jγω)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "ac_sweep.png"); plt.close(fig)
save_results("ac_sweep")

# ----------------------------------------------------------------------------------------------
# 4. Transients at Omega = 0.1, 1, 3: phase lag and power, measured on the steady-state window
# ----------------------------------------------------------------------------------------------
def fit_sinusoid(t, y, f):
    """Least-squares fit y = a cos(wt) + b sin(wt) + c; returns amplitude and phase of y = A cos(wt + phi)."""
    w = 2 * np.pi * f
    M = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    a, b, c = np.linalg.lstsq(M, y, rcond=None)[0]
    return np.hypot(a, b), np.degrees(np.arctan2(-b, a))      # A cos(wt+phi) = A cos phi cos wt - A sin phi sin wt


tran = {}
RESULTS["transients"] = {}
for key, (cir, txt, fd, t_settle, t_stop) in TRAN_FILES.items():
    hdr, d = read_wrdata(txt)
    t_raw = d[:, 0]
    period = 1.0 / fd
    # resample the last 4 periods uniformly (ngspice writes its internal time points)
    t = np.linspace(t_stop - 4 * period, t_stop, 4 * 400 + 1)
    cols = {name: np.interp(t, t_raw, d[:, i + 1]) for i, name in enumerate(["v_in", "v_cap", "i", "p_r", "p_src"])}
    q = C * cols["v_cap"]
    A_v, ph_v = fit_sinusoid(t, cols["v_in"], fd)
    A_q, ph_q = fit_sinusoid(t, q, fd)
    A_i, ph_i = fit_sinusoid(t, cols["i"], fd)
    lag = (ph_v - ph_q) % 360.0
    lead_i = (ph_i - ph_v + 180.0) % 360.0 - 180.0          # current phase relative to V: +90 = leads (capacitive)
    P_r = cols["p_r"].mean(); P_src = cols["p_src"].mean()
    Om_d = fd / F0
    lag_an = float(phase_lag_deg(Om_d, GAMMA_HAT))
    P_an = float(rlc_absorbed_power(fd, L, R, C, V0))
    A_q_an = V0 * np.abs(rlc_transfer(fd, L, R, C))
    log(f"Transient Ω={Om_d:g} ({fd:g} Hz, {len(t_raw)} SPICE points): q lags V by {lag:.3f}° (analytic {lag_an:.3f}°); "
        f"i leads V by {lead_i:+.2f}° (analytic {90 - lag_an:+.2f}°); |q| = {A_q*1e6:.4f} µC (analytic {A_q_an*1e6:.4f}, "
        f"{(A_q/A_q_an-1)*100:+.2f} %); <P_R> = {P_r*1e3:.4f} mW, <P_src> = {P_src*1e3:.4f} mW, analytic {P_an*1e3:.4f} mW "
        f"({(P_r/P_an-1)*100:+.2f} %)")
    tran[key] = dict(t=t, fd=fd, Om=Om_d, **cols, q=q, lag=lag, lag_an=lag_an, P_r=P_r, P_src=P_src, P_an=P_an)
    RESULTS["transients"][key] = {
        "f_drive_Hz": fd, "Omega": Om_d, "spice_points": int(len(t_raw)),
        "phase_lag_spice_deg": float(lag), "phase_lag_analytic_deg": lag_an,
        "current_lead_spice_deg": float(lead_i), "current_lead_analytic_deg": float(90 - lag_an),
        "q_amplitude_spice_C": float(A_q), "q_amplitude_analytic_C": float(A_q_an),
        "mean_resistor_power_spice_W": float(P_r), "mean_source_power_spice_W": float(P_src),
        "mean_power_analytic_W": P_an,
    }

# Figure 2: three columns (low, res, high); rows: drive & charge, current & drive, instantaneous power
fig, axs = plt.subplots(3, 3, figsize=(13, 9), sharex="col")
titles = {"low": "Ω = 0.1 (quasi-static): q follows V in phase",
          "res": "Ω = 1 (resonance): q lags V by 90°, current in phase",
          "high": "Ω = 3 (above resonance): q is 180° out of phase"}
def headroom(ax, frac):
    """Extend the top of the y range by frac of the data range so legend and annotation sit above the traces."""
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + frac * (hi - lo))


for j, key in enumerate(["low", "res", "high"]):
    d = tran[key]
    tt = (d["t"] - d["t"][0]) * d["fd"]          # time in drive periods
    ax = axs[0, j]
    ax.plot(tt, d["v_in"] / V0, color=SERIES[0], label="drive V/V₀ (≡ −q_e E)")
    ax.plot(tt, d["q"] / (C * V0), color=SERIES[1], label="charge q_C/(C V₀) (≡ x/x_static)")
    ax.set_title(titles[key], fontsize=10)
    ax.set_ylabel("normalised (–)")
    headroom(ax, 0.32)
    ax.legend(loc="upper right", fontsize=7)
    ax.text(0.02, 0.96, f"lag: SPICE {d['lag']:.1f}°\nanalytic {d['lag_an']:.1f}°", transform=ax.transAxes, fontsize=8,
            va="top")
    ax = axs[1, j]
    ax.plot(tt, d["v_in"] / V0, color=SERIES[0], label="drive V/V₀")
    ax.plot(tt, d["i"] * R / V0, color=SERIES[2], label="current i·R/V₀ (≡ velocity)")
    ax.set_ylabel("normalised (–)")
    headroom(ax, 0.32)
    ax.legend(loc="upper right", fontsize=7)
    ax = axs[2, j]
    unit = 1e6 if max(abs(d["p_src"]).max(), d["p_r"].max()) < 1e-4 else 1e3
    uname = "µW" if unit == 1e6 else "mW"
    ax.plot(tt, d["p_src"] * unit, color=SERIES[0], label="V·i (from source)")
    ax.plot(tt, d["p_r"] * unit, color=SERIES[3], label="R·i² (into resistor)")
    ax.axhline(d["P_r"] * unit, color=SERIES[3], ls="--", lw=1, label="⟨R·i²⟩")
    ax.set_xlabel("time (drive periods)")
    ax.set_ylabel(f"power ({uname})")
    headroom(ax, 0.42)
    ax.text(0.02, 0.96, f"⟨V·i⟩ = {fmt_power(d['P_src'])}\n⟨R·i²⟩ = {fmt_power(d['P_r'])}\nanalytic {fmt_power(d['P_an'])}",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(loc="upper right", fontsize=7)
fig.suptitle("ngspice transients: the phase of the displacement decides how much power the drive can deliver "
             "(in phase → almost none; 90° → maximum)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "transients.png"); plt.close(fig)
save_results("transients")

# ----------------------------------------------------------------------------------------------
# 5. The power argument as a picture: phasor diagrams and P(Omega) with the phase overlaid
# ----------------------------------------------------------------------------------------------
Om_fine = np.logspace(-1.3, 1.0, 800)
Pn = np.abs(chi_hat(Om_fine, GAMMA_HAT))**2 * GAMMA_HAT * Om_fine**2   # ∝ omega * (-Im chi)
fig = plt.figure(figsize=(14, 5.4))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.6], height_ratios=[1.5, 1.0])
for j, key in enumerate(["low", "res", "high"]):
    d = tran[key]
    ax = fig.add_subplot(gs[:, j])
    lag = np.radians(d["lag"])
    Aq = abs(chi_hat(d["Om"], GAMMA_HAT)); Ai = d["Om"] * Aq
    # unit-length arrows: the panel shows DIRECTIONS only (|q| spans 0.12 to 10 across the three panels, so true
    # lengths cannot share one axis); the true lengths are written in the legend, which sits below the circle
    phasor_axes(ax, "Re (phasor direction at t = 0)", "Im")
    for ang, col, ls in [(0.0, SERIES[0], "-"), (-lag, SERIES[1], "-"), (-lag + np.pi / 2, SERIES[2], "--")]:
        ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2.5, ls=ls))
    ax.plot([], [], color=SERIES[0], label="V (≡ E): |V| = 1")
    ax.plot([], [], color=SERIES[1], label=f"q (≡ p): |q| = |χ̂| = {Aq:.2f}")
    ax.plot([], [], color=SERIES[2], ls="--", label=f"i = jωq (≡ velocity): |i| = Ω|χ̂| = {Ai:.2f}")
    ax.legend(loc="lower left", fontsize=7, title="unit arrows: directions only", title_fontsize=7)
    ax.set_title(f"Ω = {d['Om']:g}: lag {d['lag']:.0f}°, cos∠(V,i) = {np.cos(np.radians(d['lag'] - 90)):.2f}", fontsize=9)
axP = fig.add_subplot(gs[0, 3])
axP.semilogx(Om_fine, Pn / Pn.max(), color=SERIES[3], label="⟨P_abs⟩ ∝ ω·(−Im χ) = ½|V||i|cos∠(V,i)")
for key in tran:
    axP.plot(tran[key]["Om"], tran[key]["P_r"] / rlc_absorbed_power(F0, L, R, C, V0), "o", color=SERIES[3], ms=7)
axP.plot([], [], "o", color=SERIES[3], label="ngspice ⟨R i²⟩ from the transients")
axP.set_ylim(-0.05, 1.45); axP.tick_params(labelbottom=False)
axP.set_ylabel("absorbed power, normalised (–)")
axP.set_title("Absorption is maximal exactly where the lag is 90°", fontsize=10)
axP.legend(loc="upper left", fontsize=8)
axL = fig.add_subplot(gs[1, 3], sharex=axP)
axL.semilogx(Om_fine, phase_lag_deg(Om_fine, GAMMA_HAT), color=SERIES[1], label="phase lag of q behind V")
for key in tran:
    axL.plot(tran[key]["Om"], tran[key]["lag"], "o", color=SERIES[1], ms=7)
axL.plot([], [], "o", color=SERIES[1], label="ngspice transient lags")
axL.axhline(90, color=PALETTE["muted"], lw=0.8, ls="--")
axL.set_yticks([0, 90, 180]); axL.set_ylim(-10, 190)
axL.set_xlabel("Ω = ω/ω₀ (–)"); axL.set_ylabel("lag (deg)")
axL.legend(loc="center left", fontsize=8)
fig.suptitle("The 'in phase = no absorption, 90° = maximum absorption' argument: the velocity i = jωq is in phase "
             "with the drive only at resonance", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "phasor_power.png"); plt.close(fig)
save_results("phasor_power")

# ----------------------------------------------------------------------------------------------
# 6. Dimensionless mapping to silica's UV resonance; n' and n'' from chi (notes 6, 8, 9)
# ----------------------------------------------------------------------------------------------
lam0 = REF.lambda_nm * 1e-9
w_1310 = 2 * np.pi * C0 / lam0
B_u, lam_u_um, w_u = effective_uv_oscillator()
Om_1310 = w_1310 / w_u
f_circuit_1310 = Om_1310 * F0
log(f"Silica effective UV oscillator: B_u = {B_u:.4f}, lambda_u = {lam_u_um*1e3:.1f} nm, w_u = {w_u:.3e} rad/s")
log(f"  1310 nm: w = {w_1310:.3e} rad/s -> Omega = {Om_1310:.4f}  ->  circuit frequency {f_circuit_1310:.1f} Hz")
chi_1310_single = B_u * chi_hat(Om_1310, 0.0)
n_1310_single = np.sqrt(1 + chi_1310_single.real)
# three-oscillator (full Malitson) Lorentz sum with gamma -> 0: must reproduce Sellmeier exactly
A_i, w_i = malitson_oscillators()
chi_3 = lorentz_sum(np.array([w_1310]), A_i, w_i, [0, 0, 0])[0]
n_1310_3 = np.sqrt(1 + chi_3.real)
n_1310_sell = float(sellmeier_n(REF.lambda_nm * 1e-3))
log(f"  n(1310) single effective UV oscillator: {n_1310_single:.4f}; 3-oscillator Lorentz sum: {n_1310_3:.4f}; "
    f"Malitson Sellmeier: {n_1310_sell:.4f}  (textbook 1.4468)")
# quasi-static limit check (notes section 8): chi(1310)/chi(0) for the UV term
ratio_qs = float(np.abs(chi_hat(Om_1310, 0.0)))
log(f"  at 1310 nm the UV oscillator's response is {ratio_qs:.4f} x its static value (notes 8: quasi-static but not zero)")
# dispersion: which resonance sets the slope dn/dlambda at 1310 nm? (notes 10)
terms, _ = sellmeier_dn_dlam_terms(REF.lambda_nm * 1e-3)
dn_dlam_sell = float(terms.sum())
dn_dlam_uv = float(terms[:2].sum()); dn_dlam_ir = float(terms[2])
lam_grid = np.linspace(1.30, 1.32, 5) * 1e-6
w_grid = 2 * np.pi * C0 / lam_grid
n_single_grid = np.sqrt(1 + B_u * chi_hat(w_grid / w_u, 0.0).real)
dn_dlam_single = float(np.gradient(n_single_grid, lam_grid)[2] * 1e-6)
ng_sell = n_1310_sell - 1.31 * dn_dlam_sell
ng_single = n_1310_single - 1.31 * dn_dlam_single
log(f"  dn/dlambda at 1310 nm: Sellmeier {dn_dlam_sell:.4e} /um = UV terms {dn_dlam_uv:.4e} + IR term {dn_dlam_ir:.4e} "
    f"(IR share {dn_dlam_ir/dn_dlam_sell*100:.0f} %); single UV oscillator alone {dn_dlam_single:.4e} /um")
log(f"  group index n_g = n - lambda dn/dlambda: Sellmeier {ng_sell:.4f}, single UV oscillator {ng_single:.4f}")
# what n'' silica actually has (0.3 dB/km fibre at 1310): n'' = alpha lambda/(4 pi)
alpha_fibre = 0.3 / 4.343 / 1e3                    # 1/m
npp_silica = alpha_fibre * lam0 / (4 * np.pi)
# circuit's Gamma applied to silica at 1310: n'' would be
chi_c = B_u * chi_hat(Om_1310, GAMMA_HAT)
n_c = n_complex_from_chi(chi_c)
alpha_c_db_cm = 4 * np.pi * (-n_c.imag) / lam0 * 4.343 / 100
log(f"  silica n'' at 1310 nm from 0.3 dB/km: {npp_silica:.2e}; if silica had the circuit's Gamma = 0.1: n'' = {-n_c.imag:.2e} "
    f"(= {alpha_c_db_cm:.0f} dB/cm)")
# the damping that would reproduce silica's real n'' at 1310 nm, and the phase lag it implies (far below resonance
# n'' = B_u Gamma Omega / (2 n' (1-Omega^2)^2), so Gamma_silica = 2 n' n'' (1-Omega^2)^2 / (B_u Omega))
Gamma_silica = 2 * n_1310_single * npp_silica * (1 - Om_1310**2)**2 / (B_u * Om_1310)
lag_1310_circuit = float(phase_lag_deg(Om_1310, GAMMA_HAT))
lag_1310_silica = float(phase_lag_deg(Om_1310, Gamma_silica))
log(f"  phase lag of the UV oscillator at 1310 nm: {lag_1310_circuit:.3f} deg with the circuit's Gamma = 0.1; with the "
    f"damping that reproduces silica's n'' (Gamma = {Gamma_silica:.1e}) it is {lag_1310_silica:.1e} deg")
RESULTS["silica_mapping"] = {
    "B_u": float(B_u), "lambda_u_nm": float(lam_u_um * 1e3), "w_u_rad_s": float(w_u),
    "w_1310_rad_s": float(w_1310), "Omega_1310": float(Om_1310), "circuit_frequency_for_1310_Hz": float(f_circuit_1310),
    "n_1310_single_uv_oscillator": float(n_1310_single), "n_1310_three_oscillator_lorentz": float(n_1310_3),
    "n_1310_malitson_sellmeier": n_1310_sell, "n_1310_textbook": 1.4468,
    "chi_over_chi_static_at_1310": ratio_qs,
    "dn_dlambda_sellmeier_per_um": dn_dlam_sell, "dn_dlambda_uv_terms_per_um": dn_dlam_uv,
    "dn_dlambda_ir_term_per_um": dn_dlam_ir, "ir_share_of_slope_pct": float(dn_dlam_ir / dn_dlam_sell * 100),
    "dn_dlambda_single_per_um": dn_dlam_single,
    "ng_1310_sellmeier": float(ng_sell), "ng_1310_single_uv": float(ng_single),
    "n_double_prime_silica_1310_from_0p3dB_km": float(npp_silica),
    "n_double_prime_if_Gamma_0p1": float(-n_c.imag), "alpha_db_cm_if_Gamma_0p1": float(alpha_c_db_cm),
    "phase_lag_at_1310_with_circuit_Gamma_deg": lag_1310_circuit,
    "Gamma_equivalent_to_silica_n_double_prime": float(Gamma_silica),
    "phase_lag_at_1310_with_silica_damping_deg": lag_1310_silica,
}

# Figure 3: n', n'' versus frequency for a single UV oscillator with three dampings (the circuit's Gamma, /5, /20)
Om_n = np.logspace(-2, 1, 3000)
DAMPINGS = [GAMMA_HAT, GAMMA_HAT / 5, GAMMA_HAT / 20]
RESULTS["silica_mapping"]["refractive_index_dampings"] = [float(g) for g in DAMPINGS]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
ax = axs[0]
for k, G in enumerate(DAMPINGS):
    n = n_complex_from_chi(B_u * chi_hat(Om_n, G))
    ax.semilogx(Om_n, n.real, color=SERIES[k], label=f"n′, Γ = {G:g}")
ax.axhline(np.sqrt(1 + B_u), color=PALETTE["muted"], lw=0.8, ls=":")
ax.axvline(Om_1310, color=PALETTE["ink2"], lw=1, ls="--")
ax.plot(Om_1310, n_1310_single, "o", color=PALETTE["ink"], zorder=5)
ax.annotate(f"1310 nm: Ω = {Om_1310:.3f}\nn′ = {n_1310_single:.4f}", (Om_1310, n_1310_single), xytext=(0.012, 1.9), fontsize=9,
            arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"]))
ax.set_ylim(0, 3.2)
ax.set_xlabel("Ω = ω/ω_u (–)      [silica: ω_u = %.2e rad/s, λ_u = %.0f nm;  circuit: f = Ω × 1 kHz]" % (w_u, lam_u_um * 1e3))
ax.set_ylabel("real index n′ = Re√(1+χ)  (–)")
wrap_title(ax, "Far below the UV resonance the index is nearly flat (χ ≈ A/ω_u²) but rising with Ω: normal dispersion", 70)
ax.legend(loc="upper left", fontsize=8)
ax = axs[1]
for k, G in enumerate(DAMPINGS):
    n = n_complex_from_chi(B_u * chi_hat(Om_n, G))
    ax.loglog(Om_n, np.maximum(-n.imag, 1e-12), color=SERIES[k], label=f"n″, Γ = {G:g}")
ax.axvline(Om_1310, color=PALETTE["ink2"], lw=1, ls="--")
ax.set_ylim(1e-6, 3)
ax.set_xlabel("Ω = ω/ω_u (–)")
ax.set_ylabel("absorption index n″ = −Im√(1+χ)  (–)")
wrap_title(ax, f"n″ is a line of width Γ; at Ω = {Om_1310:.3f} it is ∝ ΓΩ, so transparency needs a tiny Γ", 70)
ax.text(0.03, 0.06, f"real silica at 1310 nm: n″ ≈ {npp_silica:.0e} (0.3 dB/km)\ncircuit's Γ = 0.1 would give n″ = {-n_c.imag:.1e} "
        f"({alpha_c_db_cm:.0f} dB/cm)", transform=ax.transAxes, fontsize=8)
ax.legend(loc="upper left", fontsize=8)
fig.suptitle("Complex refractive index n = n′ − j n″ from the same χ̂(Ω) that ngspice measured, scaled to silica's UV resonance "
             "(B_u = %.3f)" % B_u, fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "refractive_index.png"); plt.close(fig)

# Figure 4: n(lambda) in the transparent window: single effective oscillator vs the 3-term Lorentz sum vs Malitson
lam_um = np.linspace(0.4, 2.0, 400)
w_lam = 2 * np.pi * C0 / (lam_um * 1e-6)
n_single = np.sqrt(1 + B_u * chi_hat(w_lam / w_u, 0.0).real)
n_three = np.sqrt(1 + lorentz_sum(w_lam, A_i, w_i, [0, 0, 0]).real)
n_sell = sellmeier_n(lam_um)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(lam_um, n_sell, color=SERIES[0], lw=3, label="Malitson Sellmeier (fit to measurement)")
ax.plot(lam_um, n_three, color=SERIES[1], ls="--", label="3 Lorentz oscillators, γ→0 (2 UV + 1 IR)")
ax.plot(lam_um, n_single, color=SERIES[2], ls=":", label="1 effective UV oscillator (no IR term)")
ax.axvline(1.31, color=PALETTE["muted"], lw=0.8, ls="--"); ax.text(1.32, 1.47, "1310 nm", fontsize=8)
ax.set_xlabel("vacuum wavelength λ₀ (µm)"); ax.set_ylabel("refractive index n (–)")
wrap_title(ax, "Sellmeier IS the Lorentz oscillator with damping dropped: three resonances reproduce silica's n(λ); "
               f"the IR term supplies {dn_dlam_ir/dn_dlam_sell*100:.0f} % of the slope at 1310 nm", 80, 10)
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "silica_n_lambda.png"); plt.close(fig)
RESULTS["silica_mapping"]["max_abs_diff_three_osc_vs_sellmeier"] = float(np.max(np.abs(n_three - n_sell)))
save_results("silica_mapping")

# ----------------------------------------------------------------------------------------------
# 7. Video: sweep the drive frequency, watch the q phasor fall behind V and the absorbed power rise
# ----------------------------------------------------------------------------------------------
N_FR = 30 * 12                      # 12 s at 30 fps (the render is ~0.2 s per frame and dominates the run time)
Om_path = np.logspace(np.log10(0.1), np.log10(4.0), N_FR)
Om_path[np.argmin(np.abs(Om_path - 1.0))] = 1.0      # snap the nearest frame (0.07 % away) onto exact resonance
tt = np.linspace(0, 3, 400)
P_UNIT = V0**2 * W0 * C        # power unit: V·i / (V0^2 w0 C); at resonance <V·i> = Q/2 in these units = 1/2 V0^2/R


def frame_state(Omk):
    """Honest, unclipped traces:  q/(C V0) = |chi_hat| cos(wt - lag)  and  V·i/(V0^2 w0 C) = Omega |chi_hat| cos(wt) cos(wt - lag + 90 deg).

    The trace axis is rescaled every frame to +-1.5 * max(1, |q|, |i|): the traces are never clipped and the top
    third of the axis stays free for the legend."""
    ch = chi_hat(Omk, GAMMA_HAT); A = abs(ch)
    lag = np.arctan2(GAMMA_HAT * Omk, 1 - Omk**2)
    q_t = A * np.cos(2 * np.pi * tt - lag)
    p_t = Omk * A * np.cos(2 * np.pi * tt) * np.cos(2 * np.pi * tt - lag + np.pi / 2)
    p_mean = 0.5 * Omk * A * np.sin(lag)                # exact mean of p_t: 1/2 |V||i| cos(angle(V, i))
    ylim = 1.5 * max(1.0, A, Omk * A)
    Pk = float(np.interp(Omk, Om_fine, Pn / Pn.max()))
    return ch, lag, q_t, p_t, p_mean, ylim, Pk


FRAME_GS = dict(width_ratios=[1, 1.5, 1.2], height_ratios=[1.35, 1.0], hspace=0.12, wspace=0.34)
TRACE_LABELS = ["V/V₀", "q/(C V₀)", "V·i/(V₀²ω₀C)", "⟨V·i⟩ (mean)"]


def build_frame_axes(fig, spec=None):
    """One video frame: unit phasors (directions only) | time traces | absorbed power over lag (stacked, shared x).

    With spec (a SubplotSpec) the frame is drawn inside that cell, which is how the contact sheet stacks five of them.
    Without spec the gridspec carries explicit figure margins (no tight_layout: the aspect-equal phasor panel and the
    two-line per-frame title are not compatible with it)."""
    if spec is not None:
        gs = spec.subgridspec(2, 3, **FRAME_GS)
    else:
        gs = fig.add_gridspec(2, 3, left=0.06, right=0.985, top=0.80, bottom=0.12, **FRAME_GS)
    axp = fig.add_subplot(gs[:, 0]); axt = fig.add_subplot(gs[:, 1])
    axr = fig.add_subplot(gs[0, 2]); axl = fig.add_subplot(gs[1, 2], sharex=axr)
    phasor_axes(axp, "Re", "Im"); axp.set_title("phasor directions (rotating at ω)", fontsize=9)
    axp.plot([], [], color=SERIES[0], label="V ≡ E"); axp.plot([], [], color=SERIES[1], label="q ≡ p")
    axp.plot([], [], color=SERIES[2], ls="--", label="i = jωq ≡ velocity")
    axp.legend(loc="lower left", fontsize=7, title="unit arrows; |q| in the trace title", title_fontsize=7)
    axt.set_xlabel("time (drive periods)"); axt.set_ylabel("V/V₀,  q/(C V₀),  V·i/(V₀²ω₀C)   (–)")
    axt.axhline(0, color=PALETTE["muted"], lw=0.6)
    axr.semilogx(Om_fine, Pn / Pn.max(), color=SERIES[3], label="⟨P_abs⟩ (norm.)")
    axr.set_ylim(-0.05, 1.2); axr.tick_params(labelbottom=False)
    axr.set_ylabel("⟨P_abs⟩ (norm.)"); axr.set_title("where we are on the sweep", fontsize=9)
    axl.semilogx(Om_fine, phase_lag_deg(Om_fine, GAMMA_HAT), color=SERIES[1], label="lag")
    axl.axhline(90, color=PALETTE["muted"], lw=0.6, ls="--")
    axl.set_yticks([0, 90, 180]); axl.set_ylim(-10, 190)
    axl.set_xlabel("Ω = ω/ω₀"); axl.set_ylabel("lag of q (deg)")
    return axp, axt, axr, axl


def frame_title(Omk, ch, lag, p_mean, Pk):
    """Two short lines, so the trace-axis title never runs into the neighbouring panel's title."""
    return (f"Ω = {Omk:.2f} ({Omk*F0:.0f} Hz): |q| = {abs(ch):.2f}×static, lag {np.degrees(lag):.0f}°\n"
            f"⟨V·i⟩ = {fmt_power(p_mean*P_UNIT)} ({Pk:.2f} × max)")


fig = plt.figure(figsize=(11, 4.8))
axp, axt, axr, axl = build_frame_axes(fig)
arrV = axp.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=2.5))
arrQ = axp.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=2.5))
arrI = axp.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SERIES[2], lw=2, ls="--"))
lV, = axt.plot(tt, np.cos(2 * np.pi * tt), color=SERIES[0], label=TRACE_LABELS[0])
lQ, = axt.plot(tt, np.cos(2 * np.pi * tt), color=SERIES[1], label=TRACE_LABELS[1])
lP, = axt.plot(tt, np.cos(2 * np.pi * tt), color=SERIES[3], lw=1.2, label=TRACE_LABELS[2])
lPm = axt.axhline(0, color=SERIES[3], lw=1, ls="--", label=TRACE_LABELS[3])
axt.legend(loc="upper center", fontsize=7, ncol=2); ttl = axt.set_title("", fontsize=8.5)
dot1, = axr.plot([Om_path[0]], [0], "o", color=SERIES[3]); dot2, = axl.plot([Om_path[0]], [0], "o", color=SERIES[1])
fig.suptitle(f"Driven oscillator ≡ series RLC (Γ = {GAMMA_HAT:g}): sweeping the drive frequency through resonance "
             "(trace axis rescales with Ω)", fontsize=10)


def draw(k):
    Omk = Om_path[k]
    ch, lag, q_t, p_t, p_mean, ylim, Pk = frame_state(Omk)
    rot = np.exp(1j * (2 * np.pi * k / 60))    # slow common rotation so the "rotating phasor" is visible
    v = rot; q = np.exp(-1j * lag) * rot; i = np.exp(-1j * lag + 1j * np.pi / 2) * rot
    arrV.xy = (v.real, v.imag); arrQ.xy = (q.real, q.imag); arrI.xy = (i.real, i.imag)
    lQ.set_ydata(q_t); lP.set_ydata(p_t); lPm.set_ydata([p_mean, p_mean])
    axt.set_ylim(-ylim, ylim)
    dot1.set_data([Omk], [Pk]); dot2.set_data([Omk], [np.degrees(lag)])
    ttl.set_text(frame_title(Omk, ch, lag, p_mean, Pk))
    return arrV, arrQ, arrI, lQ, lP, lPm, dot1, dot2, ttl


t0 = time.time()
anim = FuncAnimation(fig, draw, frames=N_FR, blit=False)
anim.save(OUT / "frequency_sweep.mp4", writer=FFMpegWriter(fps=30, bitrate=2500))
plt.close(fig)
t_video = time.time() - t0
log(f"Video frequency_sweep.mp4: {N_FR} frames at 30 fps ({N_FR/30:.0f} s), rendered in {t_video:.1f} s")
RESULTS["video"] = {"frames": N_FR, "fps": 30, "duration_s": N_FR / 30, "render_s": float(t_video)}
# contact sheet: five frames chosen explicitly (Omega = 0.1, 0.5, 1, 2, 4) so the resonance frame is always shown,
# each drawn directly into its own row of axes with the same drawing code (full width)
STILL_OMEGAS = [0.1, 0.5, 1.0, 2.0, 4.0]
frames_idx = [int(np.argmin(np.abs(Om_path - x))) for x in STILL_OMEGAS]
RESULTS["video"]["still_frames"] = frames_idx
RESULTS["video"]["still_Omegas"] = [float(Om_path[k]) for k in frames_idx]
sheet_fig = plt.figure(figsize=(11, 4.6 * 5))
outer = sheet_fig.add_gridspec(5, 1, hspace=0.55, left=0.07, right=0.985, top=0.945, bottom=0.03)
for row, k in enumerate(frames_idx):
    p2, t2, r2, l2 = build_frame_axes(sheet_fig, outer[row])
    Omk = Om_path[k]
    ch, lag, q_t, p_t, p_mean, ylim, Pk = frame_state(Omk)
    for ang, col, ls in [(0.0, SERIES[0], "-"), (-lag, SERIES[1], "-"), (-lag + np.pi / 2, SERIES[2], "--")]:
        p2.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2.5, ls=ls))
    t2.plot(tt, np.cos(2 * np.pi * tt), color=SERIES[0], label=TRACE_LABELS[0])
    t2.plot(tt, q_t, color=SERIES[1], label=TRACE_LABELS[1])
    t2.plot(tt, p_t, color=SERIES[3], lw=1.2, label=TRACE_LABELS[2])
    t2.axhline(p_mean, color=SERIES[3], lw=1, ls="--", label=TRACE_LABELS[3])
    t2.set_ylim(-ylim, ylim); t2.legend(loc="upper center", fontsize=7, ncol=2)
    t2.set_title(frame_title(Omk, ch, lag, p_mean, Pk), fontsize=8.5)
    r2.plot([Omk], [Pk], "o", color=SERIES[3]); l2.plot([Omk], [np.degrees(lag)], "o", color=SERIES[1])
sheet_fig.suptitle("\n".join(textwrap.wrap(
    f"frequency_sweep.mp4: frames at Ω = {', '.join(f'{Om_path[k]:.2g}' for k in frames_idx)} (V blue, q orange, "
    "i green; power violet). The trace axis is rescaled per frame; the phasor arrows are unit length (directions only), "
    "the true |q| is in each trace title", 125)), fontsize=10, y=0.985)
sheet_fig.savefig(OUT / "frequency_sweep_frames.png", dpi=150); plt.close(sheet_fig)
save_results("video")

# ----------------------------------------------------------------------------------------------
# 8. Capstone connection: plasma dispersion (Drude = Lorentz with w0 = 0) and thermo-optic effect
# ----------------------------------------------------------------------------------------------
m_ce = 0.26 * M_E                                 # conductivity effective mass of electrons in Si
tau_e = 2.0e-13                                   # collision time ~ mu m*/q for mu ~ 1350 cm^2/Vs (assumption)
dN = 1e17                                          # cm^-3
dn_drude, npp_drude, dalpha_drude = drude_silicon(dN, lam0, REF.n_si, m_ce, tau_e)
dn_soref, dalpha_soref = soref_bennett_1310(dN)
# ring shift if the whole confined mode saw this uniform dN:  d lambda = lam * Gamma_conf * dn / n_g
dlam_ring_1e17 = lam0 * REF.confinement * dn_soref / REF.ng
# how much dn_eff the modulator needs for its 50 pm/V * 1.3 V swing:  d lambda = lam * dn_eff / n_g
dlam_swing = REF.mod_eff_pm_per_v * REF.swing_vpp * 1e-12
dneff_swing = dlam_swing * REF.ng / lam0
dN_equiv = dneff_swing / (REF.confinement * 6.2e-22)      # cm^-3, if the whole confined mode saw a uniform dN
swing_in_K = dlam_swing * 1e12 / REF.dlambda_dT_pm_per_K
# thermo-optic in Lorentz language: n^2 - 1 = A/w0^2 far from resonance  ->  d(n^2) = -2 (n^2-1) dw0/w0
dw0_over_w0_per_K = -REF.n_si * REF.dn_si_dT / (REF.n_si**2 - 1)
# compare with the Varshni-type shift of silicon's dominant ~3.4 eV (E1) transition: dE/dT ~ -2.7e-4 eV/K
varshni = -2.7e-4 / 3.4
log(f"Capstone: Drude dn for {dN:.0e} cm^-3 electrons at 1310 nm = {dn_drude:.3e} (Soref-Bennett {dn_soref:.3e}, "
    f"ratio {dn_drude/dn_soref:.2f}); Drude dalpha = {dalpha_drude:.3f} /cm vs Soref-Bennett {dalpha_soref:.2f} /cm "
    f"(Drude with tau = {tau_e:.0e} s underestimates FCA {dalpha_soref/dalpha_drude:.0f}x)")
log(f"  uniform {dN:.0e} cm^-3 in the whole mode would shift the ring by {dlam_ring_1e17*1e12:.1f} pm "
    f"(= {dlam_ring_1e17*1e12/REF.dlambda_dT_pm_per_K:.2f} K of thermal drift)")
log(f"  modulator swing {dlam_swing*1e12:.0f} pm -> dn_eff = {dneff_swing:.2e} -> equivalent uniform dN ~ {dN_equiv:.1e} cm^-3; "
    f"the same 65 pm is {swing_in_K:.1f} K of thermal drift")
log(f"  thermo-optic: dn/dT = {REF.dn_si_dT:.2e}/K needs d(w0)/w0 = {dw0_over_w0_per_K:.2e} per K "
    f"(bandgap-like Varshni shift of a 3.4 eV transition: {varshni:.2e} per K)")
RESULTS["capstone"] = {
    "dN_cm3": dN, "tau_e_s_assumed": tau_e,
    "drude_dn": float(dn_drude), "soref_bennett_dn_1300nm": float(dn_soref), "drude_over_soref_dn": float(dn_drude / dn_soref),
    "drude_dalpha_per_cm": float(dalpha_drude), "soref_bennett_dalpha_per_cm": float(dalpha_soref),
    "soref_over_drude_dalpha": float(dalpha_soref / dalpha_drude),
    "ring_shift_pm_for_uniform_1e17": float(dlam_ring_1e17 * 1e12),
    "ring_shift_1e17_in_K_equiv": float(dlam_ring_1e17 * 1e12 / REF.dlambda_dT_pm_per_K),
    "modulator_swing_pm": float(dlam_swing * 1e12), "dneff_for_swing": float(dneff_swing),
    "equivalent_uniform_dN_cm3": float(dN_equiv), "modulator_swing_in_K_equiv": float(swing_in_K),
    "dw0_over_w0_per_K_for_dn_dT": float(dw0_over_w0_per_K), "varshni_dE_over_E_per_K_E1": float(varshni),
    "thermal_shift_pm_per_K": REF.dlambda_dT_pm_per_K, "fwhm_pm": REF.fwhm_pm,
}

# Figure 5: bound vs free electrons in the same chi language, and the silicon numbers
Om_b = np.logspace(-1.5, 1.0, 1000)
fig = plt.figure(figsize=(12.5, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
ax = fig.add_subplot(gs[:, 0])
chb = chi_hat(Om_b, GAMMA_HAT); chf = drude_chi_hat(Om_b, GAMMA_HAT)
ax.semilogx(Om_b, chb.real, color=SERIES[0], label="bound (Lorentz, ω₀ > 0): Re χ̂")
ax.semilogx(Om_b, -chb.imag, color=SERIES[0], ls="--", label="bound: −Im χ̂")
ax.semilogx(Om_b, chf.real, color=SERIES[1], label="free (Drude, ω₀ → 0): Re χ̂ < 0")
ax.semilogx(Om_b, -chf.imag, color=SERIES[1], ls="--", label="free: −Im χ̂")
ax.axhline(0, color=PALETTE["muted"], lw=0.8)
ax.set_ylim(-6, 6); ax.set_xlabel("Ω = ω/ω₀  (Drude: ω₀ is only a scale; Γ = 0.1 for both)")
ax.set_ylabel("χ̂ (–)")
wrap_title(ax, "Remove the spring and the same oscillator becomes a free carrier: Re χ turns negative (index drops), "
               "absorption falls as 1/Ω³", 66)
ax.legend(fontsize=8, loc="lower right")
Ns = np.logspace(16, 19, 200)
dn_D, _, da_D = drude_silicon(Ns, lam0, REF.n_si, m_ce, tau_e)
dn_S, da_S = soref_bennett_1310(Ns)
# two stacked panels sharing x: the index change (dimensionless) and the absorption change (1/cm) have different units
axn = fig.add_subplot(gs[0, 1])
axn.loglog(Ns, -dn_D, color=SERIES[0], label="Drude (m* = 0.26 mₑ)")
axn.loglog(Ns, -dn_S, color=SERIES[0], ls="--", label="Soref–Bennett (empirical, 1.3 µm)")
axn.axvline(dN_equiv, color=PALETTE["ink2"], lw=1, ls=":")
axn.text(dN_equiv * 1.15, 1.5e-5, f"ΔN for the 65 pm\nmodulator swing\n≈ {dN_equiv:.1e} cm⁻³", fontsize=8)
axn.set_ylabel("index change −Δn (–)"); axn.tick_params(labelbottom=False)
wrap_title(axn, f"Silicon at 1310 nm: Drude gets Δn within {abs(dn_drude/dn_soref-1)*100:.0f} % of the empirical fit but "
                f"underestimates free-carrier absorption {dalpha_soref/dalpha_drude:.0f}×", 66)
axn.legend(fontsize=8, loc="upper left")
axa = fig.add_subplot(gs[1, 1], sharex=axn)
axa.loglog(Ns, da_D, color=SERIES[1], label=f"Drude, τ = {tau_e:.0e} s")
axa.loglog(Ns, da_S, color=SERIES[1], ls="--", label="Soref–Bennett")
axa.axvline(dN_equiv, color=PALETTE["ink2"], lw=1, ls=":")
axa.set_xlabel("electron density change ΔN (cm⁻³)"); axa.set_ylabel("absorption change Δα (1/cm)")
axa.legend(fontsize=8, loc="upper left")
fig.suptitle("Capstone link: the modulator's plasma-dispersion effect is the ω₀ → 0 limit of the χ that ngspice just measured",
             fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "bound_vs_free.png"); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 9. Write results.json, tools.json, log; then execute the notebook (results.json is rewritten afterwards)
# ----------------------------------------------------------------------------------------------
RESULTS["runtime_s_before_notebook"] = time.time() - T_START
save_results("figures")
ng_ver = subprocess.run([NGSPICE, "-v"], capture_output=True, text=True).stdout.split("\n")[1].strip("* ")
ff_ver = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.split("\n")[0].split(" Copyright")[0]
import scipy, matplotlib as _mpl, nbformat, ipywidgets
tools = [
    {"tool": "ngspice", "version": ng_ver,
     "what_it_is": "Open-source SPICE circuit simulator (Berkeley SPICE3 lineage) used for analogue circuit design: DC, AC small-signal, transient and noise analyses of netlists.",
     "used_for": f"Solving the series RLC (L = {L*1e3:.0f} mH, R = {R:.1f} ohm, C = {C*1e9:.0f} nF, f0 = 1 kHz, Q = 10) that is term-for-term the Lorentz bound-electron ODE: an AC sweep (10 Hz-100 kHz, {len(f_ac)} points) of the capacitor voltage (charge = C*v) and loop current, two more AC sweeps at Q = 2 and Q = 50 (peak |chi_hat| ~ Q check), and three transients at 100 Hz, 1 kHz, 3 kHz with instantaneous resistor power R*i^2 and source power V*i.",
     "result": f"AC: |chi| within {err_mag*100:.4f} % and phase within {err_ph:.4f} deg of chi = A/(w0^2-w^2+j*gamma*w); lag at f0 {ph_f0_spice:.3f} deg (expect 90); displacement peak {f_pk_spice:.1f} Hz vs f0*sqrt(1-Gamma^2/2) = {f_pk_an:.1f} Hz; power FWHM {fwhm_spice:.2f} Hz vs gamma/2pi = {fwhm_an:.1f} Hz. Peak |chi_hat| = {RESULTS['ac_sweep_other_Q']['Q2']['peak_chi_hat_spice']:.3f} / {peak_main:.3f} / {RESULTS['ac_sweep_other_Q']['Q50']['peak_chi_hat_spice']:.3f} for Q = 2 / 10 / 50 vs analytic 1/(Gamma sqrt(1-Gamma^2/4)) = {RESULTS['ac_sweep_other_Q']['Q2']['peak_chi_hat_analytic']:.3f} / {peak_main_an:.3f} / {RESULTS['ac_sweep_other_Q']['Q50']['peak_chi_hat_analytic']:.3f} (max |chi| error {max(RESULTS['ac_sweep_other_Q']['Q2']['max_rel_error_magnitude_pct'], RESULTS['ac_sweep_other_Q']['Q50']['max_rel_error_magnitude_pct']):.4f} %). Transients: lags {tran['low']['lag']:.2f}/{tran['res']['lag']:.2f}/{tran['high']['lag']:.2f} deg vs analytic {tran['low']['lag_an']:.2f}/{tran['res']['lag_an']:.2f}/{tran['high']['lag_an']:.2f}; mean resistor power at resonance {tran['res']['P_r']*1e3:.3f} mW vs 1/2 V0^2/R = {tran['res']['P_an']*1e3:.3f} mW.",
     "how_to_observe": "cd experiments/03_driven_electron_rlc && ../../.venv/bin/python run.py  -> out/ac_sweep.png, out/transients.png, out/phasor_power.png; netlists out/rlc_ac.cir, out/rlc_ac_Q2.cir, out/rlc_ac_Q50.cir, out/rlc_tran_{low,res,high}.cir with raw tables out/*.txt and logs out/*.log (run one by hand: /opt/homebrew/bin/ngspice -b out/rlc_ac.cir). Change Q_FACTOR, F0 or DRIVES at the top of run.py to change the damping, resonance or drive frequencies."},
    {"tool": "scipy / numpy", "version": f"scipy {scipy.__version__}, numpy {np.__version__}",
     "what_it_is": "NumPy is the array library; SciPy adds scientific routines (signal processing, optimisation, special functions) on top of it.",
     "used_for": "The analytic chi(w) and the complex index n = sqrt(1+chi) (lorentz.py), least-squares sinusoid fits that extract amplitude and phase from the SPICE transients, interpolated peak/FWHM extraction from the AC sweep, the Sellmeier-to-Lorentz conversion for silica with the per-resonance dn/dlambda split, and the Drude (free-carrier) estimate for silicon.",
     "result": f"n(1310 nm) from the 3-oscillator Lorentz sum = {n_1310_3:.4f} vs Malitson {n_1310_sell:.4f} (textbook 1.4468, agreement to {abs(n_1310_3-1.4468)/1.4468*100:.3f} %); single effective UV oscillator gives {n_1310_single:.4f}; Omega_1310 = {Om_1310:.4f} maps to {f_circuit_1310:.1f} Hz in the 1 kHz circuit; the IR resonance supplies {dn_dlam_ir/dn_dlam_sell*100:.0f} % of dn/dlambda at 1310 nm; Drude dn = {dn_drude:.2e} per 1e17 cm^-3 vs Soref-Bennett {dn_soref:.2e}.",
     "how_to_observe": "out/refractive_index.png, out/silica_n_lambda.png, out/bound_vs_free.png, out/results.json; the functions live in lorentz.py and are reused by explore.ipynb."},
    {"tool": "matplotlib (FuncAnimation + FFMpegWriter) with ffmpeg", "version": f"matplotlib {_mpl.__version__}, {ff_ver}",
     "what_it_is": "The standard Python plotting library; its animation module writes frame sequences through ffmpeg to mp4.",
     "used_for": f"All figures and a {N_FR/30:.0f} s video ({N_FR} frames at 30 fps) sweeping the drive frequency from Omega = 0.1 to 4 while showing the V, q and i phasor directions (unit arrows), the unclipped time traces of V/V0, q/(C V0) and V*i/(V0^2 w0 C) on an axis rescaled every frame, and the position on the stacked absorbed-power and phase-lag curves.",
     "result": f"out/frequency_sweep.mp4 (rendered in {t_video:.0f} s, most of the run time) and the contact sheet out/frequency_sweep_frames.png (five frames chosen at Omega = {', '.join(f'{Om_path[k]:.2g}' for k in frames_idx)}, drawn at full width, 150 dpi): the q phasor rotates from in-phase (power averages to ~0) to 90 deg behind at the resonance frame (|q| = {abs(chi_hat(1.0, GAMMA_HAT)):.0f}x static, power always positive, mean 1/2 V0^2/R = {tran['res']['P_an']*1e3:.2f} mW) to 180 deg (small and out of phase).",
     "how_to_observe": "open out/frequency_sweep.mp4; edit Om_path, N_FR or GAMMA_HAT in run.py to change the sweep range, length or damping."},
    {"tool": "Jupyter notebook (nbformat + nbconvert, ipywidgets)", "version": f"nbformat {nbformat.__version__}, ipywidgets {ipywidgets.__version__}, kernel photonics-sims",
     "what_it_is": "Jupyter notebooks mix code, output and text; ipywidgets adds sliders that re-run a plotting function interactively; nbconvert executes a notebook headlessly and stores the outputs.",
     "used_for": "explore.ipynb: sliders for the damping Gamma, the drive frequency Omega and the oscillator strength B = A/w0^2 that redraw the phasor diagram, time traces, chi(Omega) and n', n''; a cell that writes a netlist for any (f0, Q) and runs ngspice from the notebook (three dampings overlaid); and a wavelength slider for the silica mapping.",
     "result": f"Executed notebook with static outputs saved (default slider positions); the interactive cells work in JupyterLab. Its Q = 2 / 10 / 50 ngspice overlay reproduces the peak |chi_hat| values that run.py stores in results.json['ac_sweep_other_Q'] ({RESULTS['ac_sweep_other_Q']['Q2']['peak_chi_hat_spice']:.2f} / {peak_main:.2f} / {RESULTS['ac_sweep_other_Q']['Q50']['peak_chi_hat_spice']:.2f}).",
     "how_to_observe": "cd experiments/03_driven_electron_rlc && ../../.venv/bin/jupyter lab explore.ipynb ; move the sliders. Re-execute headlessly with ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb (run.py does this at the end). Rebuild from source with ../../.venv/bin/python build_notebook.py."},
]
(OUT / "tools.json").write_text(json.dumps(tools, indent=2))

nb = HERE / "explore.ipynb"
if nb.exists():
    t0 = time.time()
    p = subprocess.run([str(JUPYTER), "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.kernel_name=photonics-sims", str(nb)],
                       capture_output=True, text=True, cwd=HERE)
    (OUT / "nbconvert.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode == 0:
        log(f"explore.ipynb executed in {time.time()-t0:.1f} s")
        RESULTS["notebook_executed"] = True
    else:
        log(f"WARNING: explore.ipynb failed to execute (see out/nbconvert.log): {p.stderr[-500:]}")
        RESULTS["notebook_executed"] = False
else:
    log("explore.ipynb not found; skipped")
    RESULTS["notebook_executed"] = False

log(f"Total runtime {time.time() - T_START:.1f} s")
save_results("done")
