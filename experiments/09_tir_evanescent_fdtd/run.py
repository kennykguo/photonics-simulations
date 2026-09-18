"""09_tir_evanescent_fdtd: a Gaussian beam meets a flat Si/SiO2 interface (Meep 2-D FDTD).

Run with the Meep interpreter from this directory:
    cd experiments/09_tir_evanescent_fdtd && ../../.meep/bin/python run.py

Notes sections implemented: 4 (wavevector, k_x^2 + beta^2 = n^2 k0^2), 17 (k_x = -j gamma),
24 (Snell, critical angle, evanescent field e^{-gamma x} e^{-j beta z}), 25 (time-averaged
Poynting vector: <S_x> = 0, <S_z> > 0 in the evanescent region), 27 (frustrated TIR = the seed of
the directional coupler).  Analytic expectations live in analytic.py, the Meep runs in fdtd.py.

Outputs (all regenerated from scratch in out/): figures (png), two videos (mp4) with contact
sheets, results.json (every headline number), tools.json (tool documentation).
"""
import sys, pathlib, json, time, platform
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REF, use_style, SERIES, PALETTE, C0
from common.params import k0_per_um
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
use_style()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm
import numpy as np
import scipy
import meep as mp

import analytic as an
import fdtd

# ----------------------------------------------------------------------------- parameters
LAM_NM = REF.lambda_nm                        # the ONE wavelength: drives K0, the analytic curves and (via lam=LAM
LAM = LAM_NM * 1e-3                           # um   in every fdtd task below) the Meep source frequency 1/LAM
N1, N2 = REF.n_si, REF.n_sio2
K0 = k0_per_um(LAM_NM)                        # rad/um
THETA_C = float(np.degrees(np.arcsin(N2 / N1)))
THETA_REF = float(np.degrees(np.arcsin(REF.neff / N1)))   # the strip mode's ray angle (n_eff = n1 sin theta)
TU_FS = 1e-6 / C0 * 1e15                      # one Meep time unit (um / c) in fs = 3.336 fs

RES_SHOW, RES_SWEEP, RES_CW = 60, 50, 40      # px/um: showcase fields, angle/gap sweeps, videos
RES_CHECK = [30, 40, 80]                      # px/um: flux-only spot checks of the two >2 % disagreements (Checks)
SHOW_ANGLES = [15.0, 24.5, 40.0]              # the three field panels and the video
TIR_SHOW = 40.0                               # the TIR angle used by poynting_profiles.png and the FTIR showcase
SHOW_RUN = SHOW_ANGLES + ([TIR_SHOW] if TIR_SHOW not in SHOW_ANGLES else [])   # always run it, even if SHOW_ANGLES changes
SWEEP_ANGLES = [15.0, 20.0, 22.0, 23.5, 24.5, 25.5, 27.0, 30.0, 35.0, 40.0, THETA_REF, 50.0]
DECAY_ANGLES = [27.0, 30.0, 35.0, 40.0, THETA_REF, 50.0]
FTIR_GAPS = [0.10, 0.15, 0.20, 0.30, 0.40]    # um
FTIR_ANGLES = [40.0, THETA_REF]
FIT_LO, FIT_HI = 0.05, 0.50                   # um: window for the evanescent decay fit
FTIR_SLOPE_MIN_GAMMA_GAP = 1.5                # fit the tunnelling slope only where sinh^2 -> e^{2 gamma d} (gamma*gap >= 1.5)
WORKERS = 6

RESULTS = {
    "inputs": {"lambda_nm": LAM_NM, "n1_si": N1, "n2_sio2": N2, "beam_waist_um": fdtd.W0,
               "cell_um": [fdtd.SX, fdtd.SY], "pml_um": fdtd.PML, "source_x_um": fdtd.SRC_X,
               "resolution_show": RES_SHOW, "resolution_sweep": RES_SWEEP, "resolution_video": RES_CW,
               "meep_time_unit_fs": TU_FS, "fdtd_module_default_lambda_um": fdtd.LAM,
               "note": "LAM_NM in run.py is the one wavelength: K0, the analytic curves and the lam= argument of every Meep run follow it"},
    "critical_angle_deg": THETA_C,
    "critical_angle_formula": "sin(theta_c) = n2/n1 (notes 24)",
    "theta_ref_deg": THETA_REF,
    "theta_ref_formula": "n_eff = n1 sin(theta): the strip mode's plane-wave angle for n_eff = 2.5 (notes 21, 22)",
}


def log(*a):
    print(f"[{time.time() - T_START:6.1f} s]", *a, flush=True)


def parallel(fn, tasks):
    import multiprocessing as mpc
    ctx = mpc.get_context("spawn")
    with ctx.Pool(min(WORKERS, len(tasks))) as pool:
        return pool.map(fn, tasks)


def f2(v):
    return float(v)


def task(**kw):
    """One fdtd run description; the wavelength is always the one defined above (LAM), never fdtd.py's default."""
    return dict(n1=N1, n2=N2, lam=LAM, **kw)


# ----------------------------------------------------------------------------- analysis helpers
def beam_column(r, x_at=0.03):
    """Index along y where |E_z| just inside the silica (x = x_at) is largest: the beam centre."""
    ix = int(np.argmin(np.abs(r["x"] - x_at)))
    return ix, int(np.argmax(np.abs(r["Ez"][ix, :])))


def fit_window(theta_deg):
    """Fit from 50 nm to 2.5 plane-wave decay lengths (capped at 500 nm): beyond ~3 decay lengths the
    FDTD tail sits on the ~3e-3 amplitude floor left by the truncated source line (see evanescent_decay.png)."""
    g, _ = an.gamma_evanescent(N1, N2, K0, theta_deg)
    return FIT_LO, float(min(FIT_HI, FIT_LO + 2.5 / g))


def decay_fit(r):
    """Fit |E_z(x)| along x at the beam centre with e^{-x/L}; return L (um) and the profile."""
    _, jy = beam_column(r)
    amp = np.abs(r["Ez"][:, jy])
    amp = amp / amp[np.argmin(np.abs(r["x"]))]
    lo, hi = fit_window(r["theta_deg"])
    L, _ = an.fit_decay_length(r["x"], amp, lo, hi)
    return L, amp


def analytic_decay(theta_deg):
    """Plane-wave 1/gamma and the finite-beam decay length fitted over the same window."""
    g, Lpw = an.gamma_evanescent(N1, N2, K0, theta_deg)
    spec = an.GaussianBeamSpectrum(N1, N2, K0, theta_deg, fdtd.W0)
    xx = np.linspace(0, 1.0, 401)
    lo, hi = fit_window(theta_deg)
    Lbeam, _ = an.fit_decay_length(xx, spec.decay_at_z(xx, 0.0), lo, hi)
    return float(Lpw), float(Lbeam), float(g), spec


def beam_waist_check(r0, theta_deg):
    """Fit the incident beam's footprint |E_z(y)| on the line x = 0 of the uniform-silicon run with a
    Gaussian exp(-((y - y0) cos(theta) / w)^2): checks what Meep means by beam_w0 (field 1/e radius)."""
    ix = int(np.argmin(np.abs(r0["x"])))
    amp = np.abs(r0["Ez"][ix, :]); y = r0["y"]
    m = amp > 0.2 * amp.max()
    p = np.polyfit(y[m], np.log(amp[m]), 2)            # ln a = -(cos th/w)^2 (y-y0)^2 + ...
    w = np.cos(np.radians(theta_deg)) / np.sqrt(-p[0])
    return float(w), float(-p[1] / (2 * p[0]))


def beta_fit(r, x_at=0.1, half=1.0):
    """Slope of the phase of E_z along the interface (in silica) = the shared tangential beta."""
    ix, jy = beam_column(r, x_at)
    y = r["y"]; ph = np.unwrap(np.angle(r["Ez"][ix, :]))
    m = np.abs(y - y[jy]) < half
    p = np.polyfit(y[m], ph[m], 1)
    return -p[0]


def poynting(r):
    Sx, Sy = an.poynting_2d(r["Ez"], r["Hx"], r["Hy"])
    return Sx, Sy


def line_flux(r, S, x_at):
    """Integral over y of a Poynting component on the line x = x_at (Meep units)."""
    ix = int(np.argmin(np.abs(r["x"] - x_at)))
    return float(np.trapezoid(S[ix, :], r["y"]))


def field_extent(r):
    return [r["x"][0], r["x"][-1], r["y"][0], r["y"][-1]]


def draw_interfaces(ax, gap=None, **kw):
    kw = dict(color=PALETTE["ink"], lw=1.0, ls="--", **kw)
    ax.axvline(0, **kw)
    if gap is not None:
        ax.axvline(gap, **kw)


def media_labels(ax, gap=None, y=None):
    y = ax.get_ylim()[1] - 0.35 if y is None else y
    bb = dict(boxstyle="round,pad=0.2", fc=PALETTE["surface"], ec="none", alpha=0.8)
    clad = f"{'silica' if abs(N2 - REF.n_sio2) < 1e-9 else 'cladding'} n = {N2:.2f}"
    ax.text(ax.get_xlim()[0] + 0.15, y, f"silicon n = {N1:.2f}", fontsize=8, color=PALETTE["ink2"], va="top", bbox=bb)
    if gap is None:
        ax.text(0.15, y, clad, fontsize=8, color=PALETTE["ink2"], va="top", bbox=bb)
    else:
        ax.text(gap + 0.15, y, "silicon", fontsize=8, color=PALETTE["ink2"], va="top", bbox=bb)


# =============================================================================== main
def main():
    global T_START
    T_START = time.time()
    for p in OUT.iterdir():                                        # regenerate everything (except the log being written)
        if p.is_file() and p.name != "run_log.txt":
            p.unlink()
    if abs(fdtd.LAM - LAM) > 1e-12:      # harmless: every task below passes lam=LAM explicitly; fdtd.LAM is only its standalone default
        log(f"note: run.py uses {LAM_NM:.0f} nm (LAM_NM); fdtd.py's own default LAM = {fdtd.LAM:.4f} um is only used by its standalone benchmark")
    log(f"Si/SiO2 at {LAM_NM:.0f} nm: theta_c = {THETA_C:.2f} deg; strip-mode angle for n_eff {REF.neff} = {THETA_REF:.2f} deg")

    # --------------------------------------------------------------------------- A. showcase runs
    tasks = []
    for th in SHOW_RUN:
        tasks.append(task(theta_deg=th, resolution=RES_SHOW, interface=False, dft_region=(0.0, 0.2)))
        tasks.append(task(theta_deg=th, resolution=RES_SHOW, dft_region=(0.0, 8.0)))
    # FTIR showcase (gap 200 nm at 40 deg) with full fields, and its flux plane in the second slab
    tasks.append(task(theta_deg=TIR_SHOW, resolution=RES_SHOW, gap=0.20, dft_region=(0.0, 8.0),
                      flux_x=(-0.3, 0.1, 0.5)))
    log(f"A. {len(tasks)} showcase phasor runs at {RES_SHOW} px/um on {WORKERS} workers ...")
    RESULTS["run_counts"] = {"showcase_phasor": len(tasks)}
    res = parallel(fdtd.phasor_task, tasks)
    show = {}
    RESULTS["beam_waist_check"] = {}
    for th, r0, r in zip(SHOW_RUN, res[0::2], res[1::2]):
        r["P_inc"] = r0["flux"][1]                         # Meep flux monitor: Re int E x H* dy (no 1/2)
        r["P_inc_S"] = 0.5 * r0["flux"][1]                 # the same power in the notes' 1/2 Re{E x H*} normalisation
        r["T"] = r["flux"][1] / r["P_inc"]                 # power crossing into silica / incident
        r["net_before"] = r["flux"][0] / r["P_inc"]        # net +x power in silicon just before the interface
        w, y0 = beam_waist_check(r0, th)
        RESULTS["beam_waist_check"][f"{th:g}_deg"] = {"w_fit_um": w, "w0_requested_um": fdtd.W0, "agreement_pct": 100 * (w / fdtd.W0 - 1),
                                                      "centre_y_um": y0, "focus_y_requested_um": fdtd.Y_FOCUS}
        show[th] = r
    ftir_show = res[-1]
    ftir_show["P_inc"] = show[TIR_SHOW]["P_inc"]; ftir_show["P_inc_S"] = show[TIR_SHOW]["P_inc_S"]
    log("   done; incident powers:", {th: round(show[th]["P_inc"], 2) for th in SHOW_RUN},
        "; beam waist fits (um):", {th: round(RESULTS["beam_waist_check"][f"{th:g}_deg"]["w_fit_um"], 3) for th in SHOW_RUN})

    RESULTS["showcase"] = {}
    for th in SHOW_RUN:
        r = show[th]
        spec = an.GaussianBeamSpectrum(N1, N2, K0, th, fdtd.W0)
        beta = N1 * K0 * np.sin(np.radians(th))
        _, _, Tpw = an.fresnel_s(N1, N2, K0, np.array([beta]))
        Sx, Sy = poynting(r)
        r["Sx"], r["Sy"] = Sx, Sy
        d = {
            "T_fdtd": f2(r["T"]), "T_beam_analytic": spec.transmission_beam(), "T_plane_wave_fresnel": f2(Tpw[0]),
            "T_fdtd_vs_beam_pct": 100 * (r["T"] / spec.transmission_beam() - 1) if spec.transmission_beam() > 1e-6 else None,
            "R_plus_T_fdtd": f2(1 - r["net_before"] + r["T"]),
            "beta_fit_rad_per_um": f2(beta_fit(r)), "beta_expected_rad_per_um": f2(beta),
            # field-derived 1/2 Re int(E x H*) dy at x = 0.3 vs Meep's flux monitor (which omits the 1/2): expect 0.5
            "poynting_from_fields_over_flux_monitor": f2(line_flux(r, Sx, 0.3) / r["flux"][1]) if r["T"] > 1e-2 else None,
        }
        d["beta_agreement_pct"] = 100 * (d["beta_fit_rad_per_um"] / beta - 1)
        if d["poynting_from_fields_over_flux_monitor"] is not None:
            d["poynting_vs_flux_monitor_agreement_pct"] = 100 * (2 * d["poynting_from_fields_over_flux_monitor"] - 1)
        ix = int(np.argmin(np.abs(r["x"] - 0.1)))
        # net normal / net parallel power flow 100 nm inside the silica, for EVERY angle (order 1 below
        # critical, ~1e-4 above): the number to watch when walking the angle through theta_c
        d["net_Sx_over_net_Sy_at_x0p1um"] = f2(Sx[ix].sum() / Sy[ix].sum())
        d["net_normal_power_in_silica_over_incident"] = f2(line_flux(r, Sx, 0.1) / r["P_inc_S"])
        # the same net normal power 2 um into the silica: equal to the x = 0.1 um value below and above theta_c,
        # lower at theta_c because the grazing transmitted wave leaves the monitored strip sideways (Checks)
        d["net_normal_power_in_silica_over_incident_at_x2um"] = f2(line_flux(r, Sx, 2.0) / r["P_inc_S"])
        if th > THETA_C + 1:
            Lf, _ = decay_fit(r); Lpw, Lb, g, _ = analytic_decay(th)
            d.update({
                "decay_length_fdtd_nm": 1e3 * Lf, "decay_length_plane_wave_nm": 1e3 * Lpw,
                "decay_length_beam_analytic_nm": 1e3 * Lb, "fit_window_um": list(fit_window(th)),
                "decay_vs_plane_wave_pct": 100 * (Lf / Lpw - 1), "decay_vs_beam_pct": 100 * (Lf / Lb - 1),
                "max_local_Sx_over_max_Sy_at_x0p1um": f2(np.abs(Sx[ix]).max() / Sy[ix].max()),
                "parallel_power_in_silica_over_incident_at_x0p1um": f2(line_flux(r, Sy, 0.1) / r["P_inc_S"]),
            })
        elif th <= THETA_C - 0.5:
            # direction of the power flow in silica vs Snell
            mreg = (r["x"] > 0.5) & (r["x"] < 2.5)
            ang = float(np.degrees(np.arctan2(Sy[mreg].sum(), Sx[mreg].sum())))
            snell = float(np.degrees(np.arcsin(min(1.0, N1 * np.sin(np.radians(th)) / N2))))
            d.update({"transmitted_direction_from_poynting_deg": ang, "snell_angle_deg": snell,
                      "direction_agreement_pct": 100 * (ang / snell - 1)})
        RESULTS["showcase"][f"{th:g}_deg"] = d
        log(f"   theta = {th:5.1f} deg: T_fdtd = {d['T_fdtd']:.4g}  T_beam = {d['T_beam_analytic']:.4g}  "
            f"T_pw = {d['T_plane_wave_fresnel']:.4g}  R+T = {d['R_plus_T_fdtd']:.4f}  beta {d['beta_agreement_pct']:+.2f} %")

    # --------------------------------------------------------------------------- B. angle sweep
    tasks = []
    for th in SWEEP_ANGLES:
        tasks.append(task(theta_deg=th, resolution=RES_SWEEP, interface=False))
        tasks.append(task(theta_deg=th, resolution=RES_SWEEP, dft_region=(0.5, 1.5)))
    # flux-only resolution spot checks of the two FDTD-vs-analytic gaps above 2 %: T at the critical angle
    # (24.5 deg) and the FTIR T at the strip-mode angle with a 200 nm gap (see Checks in the README)
    checks = [(24.5, None)] * len(RES_CHECK) + [(THETA_REF, 0.20)] * len(RES_CHECK)
    check_res = RES_CHECK * 2
    for (th, gap), rs in zip(checks, check_res):
        tasks.append(task(theta_deg=th, resolution=rs, interface=False))
        tasks.append(task(theta_deg=th, resolution=rs, gap=gap, flux_x=(-0.3, 0.3) if gap is None else (-0.3, gap + 0.3)))
    nsw = 2 * len(SWEEP_ANGLES)
    log(f"B. {nsw} angle-sweep runs at {RES_SWEEP} px/um + {len(tasks) - nsw} flux-only resolution checks at {RES_CHECK} px/um ...")
    RESULTS["run_counts"].update(angle_sweep_phasor=nsw, resolution_check_phasor=len(tasks) - nsw)
    res = parallel(fdtd.phasor_task, tasks)
    res, res_chk = res[:nsw], res[nsw:]
    sweep = []
    for th, r0, r in zip(SWEEP_ANGLES, res[0::2], res[1::2]):
        P = r0["flux"][1]
        spec = an.GaussianBeamSpectrum(N1, N2, K0, th, fdtd.W0)
        beta = N1 * K0 * np.sin(np.radians(th))
        _, _, Tpw = an.fresnel_s(N1, N2, K0, np.array([beta]))
        row = dict(theta_deg=th, T_fdtd=r["flux"][1] / P, R_plus_T=1 - r["flux"][0] / P + r["flux"][1] / P,
                   T_beam=spec.transmission_beam(), T_plane_wave=f2(Tpw[0]))
        if th in DECAY_ANGLES:
            Lf, prof = decay_fit(r); Lpw, Lb, g, _ = analytic_decay(th)
            row.update(decay_fdtd_nm=1e3 * Lf, decay_plane_wave_nm=1e3 * Lpw, decay_beam_nm=1e3 * Lb,
                       gamma_per_um=g, fit_window_um=list(fit_window(th)),
                       vs_plane_wave_pct=100 * (Lf / Lpw - 1), vs_beam_pct=100 * (Lf / Lb - 1))
            r["profile"] = prof
        r["row"] = row
        sweep.append((row, r))
        log(f"   theta = {th:5.2f}: T = {row['T_fdtd']:.3e} (beam {row['T_beam']:.3e}, pw {row['T_plane_wave']:.3e})"
            + (f"  1/gamma fit {row['decay_fdtd_nm']:.0f} nm vs {row['decay_plane_wave_nm']:.0f} nm" if "decay_fdtd_nm" in row else ""))
    RESULTS["angle_sweep"] = [row for row, _ in sweep]
    RESULTS["angle_sweep_note"] = ("T_fdtd = power flux through x = +0.3 um in silica / incident beam power (uniform-silicon "
                                   "normalisation run). T_beam = plane-wave Fresnel T averaged over the beam's angular spectrum "
                                   "(analytic.GaussianBeamSpectrum). Beyond critical the FDTD value is a numerical floor set by "
                                   "the truncation of the source line, not physics.")
    RESULTS["decay_fit_window_rule"] = f"ln|E_z| vs x fitted from {FIT_LO*1e3:.0f} nm to min({FIT_HI*1e3:.0f} nm, {FIT_LO*1e3:.0f} nm + 2.5/gamma) on the beam centre line"
    # numerical floor of the single-interface T well beyond critical: angles >= 35 deg if any were swept,
    # else anything beyond theta_c + 5 deg, else unknown (None: the floor line is then not drawn)
    floor_rows = [row for row, _ in sweep if row["theta_deg"] >= 35] or [row for row, _ in sweep if row["theta_deg"] >= THETA_C + 5]
    RESULTS["tir_floor_T_fdtd"] = max(row["T_fdtd"] for row in floor_rows) if floor_rows else None
    RESULTS["tir_floor_angles_deg"] = [row["theta_deg"] for row in floor_rows]

    # resolution spot checks (flux only): the same T at 30 / 40 / 80 px/um next to the sweep (50) and showcase (60) values
    RESULTS["resolution_check"] = {}
    for (th, gap), rs, r0, r in zip(checks, check_res, res_chk[0::2], res_chk[1::2]):
        key = f"{th:g}_deg" + ("" if gap is None else f"_gap{1e3*gap:.0f}nm")
        spec = an.GaussianBeamSpectrum(N1, N2, K0, th, fdtd.W0)
        Tb = spec.transmission_beam(gap)
        entry = RESULTS["resolution_check"].setdefault(key, {"theta_deg": th, "gap_nm": None if gap is None else 1e3 * gap,
                                                             "T_beam_analytic": Tb, "T_fdtd_by_resolution": {}})
        entry["T_fdtd_by_resolution"][str(rs)] = r["flux"][1] / r0["flux"][1]
    rc = RESULTS["resolution_check"]["24.5_deg"]["T_fdtd_by_resolution"]
    for row, _ in sweep:
        if row["theta_deg"] == 24.5:
            rc[str(RES_SWEEP)] = row["T_fdtd"]
    if "24.5_deg" in RESULTS["showcase"]:
        rc[str(RES_SHOW)] = RESULTS["showcase"]["24.5_deg"]["T_fdtd"]
    for key, entry in RESULTS["resolution_check"].items():
        entry["T_fdtd_vs_beam_pct_by_resolution"] = {k: 100 * (v / entry["T_beam_analytic"] - 1) for k, v in sorted(entry["T_fdtd_by_resolution"].items(), key=lambda kv: int(kv[0]))}
        log(f"   resolution check {key}: " + ", ".join(f"{k} px/um: {v:+.2f} %" for k, v in entry["T_fdtd_vs_beam_pct_by_resolution"].items()))

    # --------------------------------------------------------------------------- C. frustrated TIR sweep
    tasks = []
    for th in FTIR_ANGLES:
        tasks.append(task(theta_deg=th, resolution=RES_SWEEP, interface=False))
        for gap in FTIR_GAPS:
            tasks.append(task(theta_deg=th, resolution=RES_SWEEP, gap=gap, flux_x=(-0.3, gap + 0.3)))
    log(f"C. {len(tasks)} frustrated-TIR runs at {RES_SWEEP} px/um ...")
    RESULTS["run_counts"]["ftir_phasor"] = len(tasks)
    res = parallel(fdtd.phasor_task, tasks)
    ftir = {}
    k = 0
    for th in FTIR_ANGLES:
        P = res[k]["flux"][1]; k += 1
        rows = []
        for gap in FTIR_GAPS:
            r = res[k]; k += 1
            RESULTS["resolution_check"].get(f"{th:g}_deg_gap{1e3*gap:.0f}nm", {}).get("T_fdtd_by_resolution", {})[str(RES_SWEEP)] = r["flux"][1] / P
            spec = an.GaussianBeamSpectrum(N1, N2, K0, th, fdtd.W0)
            rows.append(dict(gap_nm=1e3 * gap, T_fdtd=r["flux"][1] / P, R_plus_T=1 - r["flux"][0] / P + r["flux"][1] / P,
                             T_plane_wave=f2(an.ftir_closed_form(N1, N2, K0, th, gap)),
                             T_beam=spec.transmission_beam(gap)))
        g, Lpw = an.gamma_evanescent(N1, N2, K0, th)
        gaps = np.array([row["gap_nm"] for row in rows]) * 1e-3
        Tf = np.array([row["T_fdtd"] for row in rows])
        # slope of ln T vs gap in the asymptotic region sinh^2(gamma d) -> e^{2 gamma d}/4, i.e. gamma*gap >= 1.5
        # (200 nm and beyond at these angles); with fewer than two such gaps fall back to the two largest gaps
        # present, and say so in results.json (the fitted slope is then below 2 gamma because sinh^2 has not
        # yet become a pure exponential)
        if len(gaps) < 2:
            raise ValueError("FTIR_GAPS needs at least two gaps for the tunnelling-slope fit")
        m = g * gaps >= FTIR_SLOPE_MIN_GAMMA_GAP
        slope_fit_asymptotic = bool(m.sum() >= 2)
        if not slope_fit_asymptotic:
            m = gaps >= np.sort(gaps)[-2]
        slope = -np.polyfit(gaps[m], np.log(Tf[m]), 1)[0]
        Tpw_same = np.array([row["T_plane_wave"] for row in rows])
        slope_pw = -np.polyfit(gaps[m], np.log(Tpw_same[m]), 1)[0]   # same gaps, closed form (sinh^2 not yet pure exp)
        ftir[th] = dict(rows=rows, gamma_per_um=f2(g), decay_length_nm=1e3 * f2(Lpw), two_gamma_per_um=2 * f2(g),
                        slope_fit_gaps_nm=[f2(1e3 * v) for v in gaps[m]], slope_fit_in_asymptotic_region=slope_fit_asymptotic,
                        fitted_slope_per_um=f2(slope), slope_agreement_pct=100 * (slope / (2 * g) - 1),
                        plane_wave_slope_same_gaps_per_um=f2(slope_pw),
                        slope_vs_plane_wave_same_gaps_pct=100 * (slope / slope_pw - 1),
                        T_fdtd_vs_beam_pct=[100 * (row["T_fdtd"] / row["T_beam"] - 1) for row in rows],
                        dT_over_T_per_10nm_pct_fdtd=100 * (np.exp(-slope * 0.01) - 1),
                        dT_over_T_per_10nm_pct_analytic=100 * (np.exp(-2 * g * 0.01) - 1),
                        factor_per_10nm_fdtd_over_analytic_pct=100 * (np.exp(-slope * 0.01) / np.exp(-2 * g * 0.01) - 1),
                        dT_over_T_per_10nm_difference_percentage_points=100 * (np.exp(-slope * 0.01) - np.exp(-2 * g * 0.01)))
        log(f"   FTIR theta = {th:.2f}: 1/gamma = {1e3*Lpw:.0f} nm, slope fit over gaps {ftir[th]['slope_fit_gaps_nm']} nm: {slope:.2f}/um vs 2 gamma {2*g:.2f}/um (closed form over the same gaps {slope_pw:.2f}/um); "
            f"T(gap) = " + ", ".join(f"{row['gap_nm']:.0f} nm: {row['T_fdtd']:.3g}" for row in rows))
    RESULTS["ftir"] = {f"{th:g}_deg": ftir[th] for th in FTIR_ANGLES}
    for key, entry in RESULTS["resolution_check"].items():          # add the FTIR sweep value, then re-sort
        entry["T_fdtd_vs_beam_pct_by_resolution"] = {k: 100 * (v / entry["T_beam_analytic"] - 1) for k, v in sorted(entry["T_fdtd_by_resolution"].items(), key=lambda kv: int(kv[0]))}
        # how the excess over the analytic value falls with resolution: power-law exponent of excess vs resolution,
        # and the excess extrapolated to infinite resolution assuming first-order (linear in 1/res) and
        # second-order (linear in 1/res^2) convergence; the true limit lies between the two
        rs = np.array([int(k) for k in entry["T_fdtd_vs_beam_pct_by_resolution"]], dtype=float)
        ex = np.array(list(entry["T_fdtd_vs_beam_pct_by_resolution"].values()))
        if len(rs) >= 3 and np.all(ex > 0):
            entry["excess_vs_resolution_power_law_exponent"] = f2(-np.polyfit(np.log(rs), np.log(ex), 1)[0])
            entry["extrapolated_excess_pct_first_order"] = f2(np.polyfit(1 / rs, ex, 1)[1])
            entry["extrapolated_excess_pct_second_order"] = f2(np.polyfit(1 / rs**2, ex, 1)[1])
            log(f"   resolution check {key}: excess ~ resolution^-{entry['excess_vs_resolution_power_law_exponent']:.2f}; extrapolated to infinite resolution "
                f"{entry['extrapolated_excess_pct_first_order']:+.1f} % (1st order) / {entry['extrapolated_excess_pct_second_order']:+.1f} % (2nd order)")
    RESULTS["ftir_showcase_gap200nm_40deg"] = {
        "T_fdtd": f2(ftir_show["flux"][2] / ftir_show["P_inc"]),
        "T_plane_wave": f2(an.ftir_closed_form(N1, N2, K0, 40.0, 0.20)),
        "T_beam": an.GaussianBeamSpectrum(N1, N2, K0, 40.0, fdtd.W0).transmission_beam(0.20),
    }

    # --------------------------------------------------------------------------- D. capstone numbers
    g_ref = K0 * np.sqrt(REF.neff**2 - N2**2)
    g_solver = K0 * np.sqrt(2.7**2 - N2**2)
    RESULTS["capstone"] = {
        "strip_neff": REF.neff, "strip_theta_deg": THETA_REF,
        "tail_decay_length_nm_neff2p5": 1e3 / g_ref, "tail_decay_length_nm_neff2p7_solver": 1e3 / g_solver,
        "kappa2_change_per_10nm_gap_pct_neff2p5": 100 * (np.exp(-2 * g_ref * 0.01) - 1),
        "kappa2_change_per_10nm_gap_pct_neff2p7": 100 * (np.exp(-2 * g_solver * 0.01) - 1),
        "kappa_field_change_per_10nm_gap_pct_neff2p5": 100 * (np.exp(-g_ref * 0.01) - 1),
        "ftir_measured_power_change_per_10nm_gap_pct_at_theta_ref": ftir[THETA_REF]["dT_over_T_per_10nm_pct_fdtd"],
        "ref_kappa2": REF.kappa2, "ref_gap_nm_textbook": 200.0,
        "statement": "kappa^2 of the ring-bus coupler scales like the tunnelled power here: e^{-2 gamma gap}. "
                     "With 1/gamma ~ 100 nm a 10 nm gap error is ~20 % in kappa^2, and anything (a heater, "
                     "a metal via) placed inside a few decay lengths of the core loads the tail.",
    }

    # --------------------------------------------------------------------------- E. numbers quoted in "Experiments to try"
    def clad(n2):
        g, L = an.gamma_evanescent(N1, n2, K0, 40.0)
        return {"n2": n2, "critical_angle_deg": float(np.degrees(np.arcsin(n2 / N1))), "decay_length_nm_at_40deg": f2(1e3 * L)}
    g_ref40, _ = an.gamma_evanescent(N1, N2, K0, 40.0)
    g_ref_1550, L_ref_1550 = an.gamma_evanescent(N1, N2, k0_per_um(1550.0), THETA_REF)
    RESULTS["experiments_to_try"] = {
        "item1_snell_angle_at_22deg": float(np.degrees(np.arcsin(N1 * np.sin(np.radians(22.0)) / N2))),
        "item1_decay_length_nm_at_27deg": f2(1e3 * an.gamma_evanescent(N1, N2, K0, 27.0)[1]),
        "item1_net_Sx_over_net_Sy_at_x0p1um_by_showcase_angle": {k: v["net_Sx_over_net_Sy_at_x0p1um"] for k, v in RESULTS["showcase"].items()},
        "item2_angular_spread_deg_by_w0": {f"{w:g}_um": float(np.degrees(2 / (N1 * K0 * w))) for w in (1.0, 2.0, 4.0)},
        "item3_cladding": {f"n2_{n2:g}": clad(n2) for n2 in (1.0, N2, 2.0)},
        "item4_gamma_gap_at_40deg_for_150nm": f2(g_ref40 * 0.15),
        "item5_decay_length_ratio_1550_over_1310": 1550.0 / 1310.0,
        "item5_decay_length_nm_at_theta_ref_1550nm": f2(1e3 * L_ref_1550),
        "item5_ftir_T_200nm_theta_ref_1550_over_1310_closed_form": f2(an.ftir_closed_form(N1, N2, k0_per_um(1550.0), THETA_REF, 0.20)
                                                                     / an.ftir_closed_form(N1, N2, K0, THETA_REF, 0.20)),
    }

    # =========================================================================== figures
    log("figures ...")
    fig_fields(show)
    fig_poynting(show)
    fig_poynting_profiles(show)
    fig_decay(sweep)
    fig_transmission(sweep)
    fig_ftir_field(ftir_show)
    fig_ftir_gap(ftir)

    # =========================================================================== videos
    log(f"videos: CW runs at {RES_CW} px/um ...")
    tasks = [task(theta_deg=th, resolution=RES_CW) for th in SHOW_ANGLES]
    tasks += [task(theta_deg=40.0, resolution=RES_CW, gap=gap) for gap in (None, 0.15, 0.30)]
    RESULTS["run_counts"]["cw_video"] = len(tasks)
    RESULTS["run_counts"]["total_meep_runs"] = sum(RESULTS["run_counts"].values())
    cw = parallel(fdtd.cw_task, tasks)
    video(cw[:3], [f"θ = {th:g}° ({'below' if th < THETA_C else 'at' if abs(th - THETA_C) < 0.5 else 'above'} θc = {THETA_C:.1f}°)" for th in SHOW_ANGLES],
          "tir_beams", "Gaussian beam on a Si/SiO₂ interface: below critical it refracts, above it only the evanescent tail enters")
    video(cw[3:], ["single interface (TIR)", "second Si slab, gap 150 nm", "second Si slab, gap 300 nm"],
          "ftir", "Frustrated TIR at θ = 40°: a second silicon slab within the evanescent tail intercepts power (a coupler)",
          gaps=[None, 0.15, 0.30])

    # =========================================================================== results + tools
    RESULTS["runtime_s"] = time.time() - T_START
    with open(OUT / "results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    write_tools()
    log(f"done in {RESULTS['runtime_s']:.0f} s")


# =============================================================================== figures
def fig_fields(show):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True, layout="constrained")
    for ax, th in zip(axs, SHOW_ANGLES):
        r = show[th]
        E = np.real(r["Ez"]).T
        m = 0.9 * np.abs(E[(r["x"][None, :] < -0.5) & (np.abs(r["y"][:, None] - fdtd.Y_FOCUS) < 3)]).max()
        # each panel normalised to 0.9 x the peak |Re E_z| of its own incident beam, so one colorbar serves all three
        im = ax.imshow(E / m, origin="lower", extent=field_extent(r), cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        draw_interfaces(ax)
        ax.grid(False)
        # incident ray and the expected transmitted / evanescent behaviour
        th_r = np.radians(th)
        ax.annotate("", xy=(0, fdtd.Y_FOCUS), xytext=(-1.6 * np.cos(th_r), fdtd.Y_FOCUS - 1.6 * np.sin(th_r)),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["ink"], lw=1.4))
        if th < THETA_C:
            tt = np.arcsin(N1 * np.sin(th_r) / N2)
            ax.annotate("", xytext=(0, fdtd.Y_FOCUS), xy=(1.6 * np.cos(tt), fdtd.Y_FOCUS + 1.6 * np.sin(tt)),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.4))
            ax.text(1.0, fdtd.Y_FOCUS - 1.2, f"Snell: θt = {np.degrees(tt):.1f}°", fontsize=8, color=PALETTE["green"])
        elif abs(th - THETA_C) < 0.5:
            ax.text(0.25, fdtd.Y_FOCUS - 1.9, "at θc: plane-wave 1/γ → ∞;\nhalf of the beam's angular\nspectrum still propagates,\ngrazing the interface", fontsize=8, color=PALETTE["green"])
        else:
            g, L = an.gamma_evanescent(N1, N2, K0, th)
            ax.text(0.25, fdtd.Y_FOCUS - 1.6, f"evanescent tail\n1/γ = {1e3*L:.0f} nm", fontsize=8, color=PALETTE["green"])
        ax.set_title(f"θ = {th:g}°: {'refracts + reflects' if th < THETA_C else 'at the critical angle' if abs(th-THETA_C)<0.5 else 'total internal reflection'}", fontsize=10)
        ax.set_xlabel("x, normal to the interface (µm)")
        media_labels(ax)
    axs[0].set_ylabel("y, along the interface (µm)")
    fig.colorbar(im, ax=axs.tolist(), fraction=0.025, pad=0.02, label="Re E_z / (0.9 × peak |Re E_z| of the incident beam)")
    fig.suptitle(f"Re E_z phasor at {LAM_NM:.0f} nm, s-polarised beam (w0 = {fdtd.W0:g} µm) from silicon; θc = {THETA_C:.1f}°", fontsize=11)
    fig.savefig(OUT / "fields_three_angles.png"); plt.close(fig)


def fig_poynting(show):
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True, layout="constrained")
    for ax, th in zip(axs, SHOW_ANGLES):
        r = show[th]; Sx, Sy = r["Sx"], r["Sy"]
        x, y = r["x"], r["y"]
        mx = (x > -1.2) & (x < 1.2); my = (y > -1.5) & (y < 2.8)
        S = np.hypot(Sx, Sy)[np.ix_(mx, my)]
        Smax = S.max()
        im = ax.imshow(S.T / Smax, origin="lower", extent=[x[mx][0], x[mx][-1], y[my][0], y[my][-1]], cmap="Blues",
                       norm=LogNorm(vmin=1e-4, vmax=1), aspect="equal")
        step = 6
        xs, ys = x[mx][::step], y[my][::step]
        U, V = Sx[np.ix_(mx, my)][::step, ::step].T, Sy[np.ix_(mx, my)][::step, ::step].T
        mag = np.hypot(U, V)
        keep = mag > 1e-4 * Smax
        Un, Vn = np.where(keep, U / np.where(mag > 0, mag, 1), np.nan), np.where(keep, V / np.where(mag > 0, mag, 1), np.nan)
        ax.quiver(xs, ys, Un, Vn, color=PALETTE["orange"], scale=28, width=0.004, headwidth=4)
        draw_interfaces(ax); ax.grid(False)
        ax.set_title(f"θ = {th:g}°: arrows = direction of ⟨S⟩", fontsize=10)
        ax.set_xlabel("x, normal to the interface (µm)")
        media_labels(ax)
    axs[0].set_ylabel("y, along the interface (µm)")
    fig.colorbar(im, ax=axs.tolist(), fraction=0.025, pad=0.02, label="|⟨S⟩| / max |⟨S⟩| in the panel (log scale)")
    fig.suptitle("Time-averaged Poynting vector ⟨S⟩ = ½ Re{E × H*}: below critical power crosses; above critical the silica-side arrows run parallel to the interface\n"
                 "(background: |⟨S⟩| on a log scale, 4 decades; arrows are unit length, direction only)", fontsize=10)
    fig.savefig(OUT / "poynting_three_angles.png"); plt.close(fig)


def fig_poynting_profiles(show):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.8))
    axs = axs.ravel()
    # (a) net normal power across the whole line x = const, vs x, normalised to P_inc
    # (b) the parallel flow on the silica side, on its own panel (its scale differs from (a))
    ax_n, ax_p = axs[0], axs[1]
    for i, th in enumerate(SHOW_ANGLES):
        r = show[th]; x = r["x"]
        Px = np.array([np.trapezoid(r["Sx"][ix, :], r["y"]) for ix in range(len(x))]) / r["P_inc_S"]
        Py = np.array([np.trapezoid(r["Sy"][ix, :], r["y"]) for ix in range(len(x))]) / r["P_inc_S"]
        ax_n.plot(x, Px, color=SERIES[i], label=f"θ = {th:g}°")
        ms = x > 0
        ax_p.plot(x[ms], Py[ms], color=SERIES[i], label=f"θ = {th:g}°")
    ax_n.set_xlim(-1.0, 2.0); ax_n.set_ylim(-0.05, 1.05)
    ax_n.axvline(0, color=PALETTE["ink"], lw=0.8, ls="--")
    ax_n.set_xlabel("x, normal to the interface (µm)"); ax_n.set_ylabel("net normal power ∫⟨S_x⟩dy / P_inc")
    ax_n.set_title("(a) Net normal power ∫⟨S_x⟩dy: ≈ T below θc, ≈ 0 above θc\n(24.5°: the grazing wave leaves the 10 µm strip sideways, so it falls with x)", fontsize=9)
    ax_n.legend(fontsize=8, loc="center left")
    ax_p.set_xlim(0, 2.0); ax_p.set_ylim(-0.05, 1.05 * max(1.0, ax_p.get_ylim()[1]))
    ax_p.set_xlabel("x into the silica (µm)"); ax_p.set_ylabel("parallel flow ∫⟨S_y⟩dy / P_inc, silica side")
    ax_p.set_title("(b) Parallel flow ∫⟨S_y⟩dy on the silica side: ≠ 0 at every angle;\nat 40° it is confined to the tail and decays with x", fontsize=9)
    ax_p.legend(fontsize=8, loc="upper right")
    # (c) local S_x(y) and S_y(y) at x = 0.1 um for 40 deg
    ax = axs[2]
    r = show[TIR_SHOW]; ix = int(np.argmin(np.abs(r["x"] - 0.1)))
    sc = r["Sy"][ix].max()
    ax.plot(r["y"], r["Sy"][ix] / sc, color=SERIES[0], label="⟨S_y⟩ (parallel to interface)")
    ax.plot(r["y"], r["Sx"][ix] / sc, color=SERIES[1], label="⟨S_x⟩ (normal)")
    ax.axhline(0, color=PALETTE["ink"], lw=0.6)
    ax.set_xlim(-2.5, 3.5)
    ax.set_xlabel("y, along the interface (µm)"); ax.set_ylabel("⟨S⟩ / max ⟨S_y⟩ at x = 0.1 µm")
    ax.set_title("(c) θ = 40°, 100 nm inside the silica: the beam's power runs along\nthe interface; ⟨S_x⟩ is a small in/out loop that nets to zero", fontsize=9)
    ax.legend(fontsize=8)
    # (d) decay of |Ez|^2, S_y along x at the beam centre for 40 deg
    ax = axs[3]
    _, jy = beam_column(r)
    x = r["x"]; m = (x > -0.6) & (x < 0.8)
    E2 = np.abs(r["Ez"][:, jy]) ** 2; sc = E2[np.argmin(np.abs(x))]
    ax.semilogy(x[m], E2[m] / sc, color=SERIES[0], label="|E_z|² (FDTD)")
    ax.semilogy(x[m], np.abs(r["Sy"][m, jy]) / r["Sy"][np.argmin(np.abs(x)), jy], color=SERIES[2], label="⟨S_y⟩ (FDTD)")
    ax.semilogy(x[m], np.abs(r["Sx"][m, jy]) / r["Sy"][np.argmin(np.abs(x)), jy], color=SERIES[1], ls=":", label="|⟨S_x⟩| (FDTD)")
    g, L = an.gamma_evanescent(N1, N2, K0, TIR_SHOW)
    xx = np.linspace(0, 0.8, 50)
    ax.semilogy(xx, np.exp(-2 * g * xx), color=PALETTE["ink"], ls="--", lw=1.2, label=f"e^(−2γx), 1/γ = {1e3*L:.0f} nm")
    ax.axvline(0, color=PALETTE["ink"], lw=0.8, ls="--")
    ax.set_ylim(1e-5, 3); ax.set_xlabel("x (µm), beam centre line"); ax.set_ylabel("normalised to the interface value")
    ax.set_title("(d) θ = 40°: intensity and parallel power decay as e^(−2γx);\nthe normal component is ~30× smaller and reactive", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "poynting_profiles.png"); plt.close(fig)


def fig_decay(sweep):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    for i, (row, r) in enumerate([s for s in sweep if s[0]["theta_deg"] in DECAY_ANGLES]):
        th = row["theta_deg"]; x = r["x"]; m = (x >= 0) & (x < 1.0)
        c = SERIES[i % len(SERIES)]
        ax.semilogy(x[m], r["profile"][m], color=c, label=f"θ = {th:.1f}°: fit {row['decay_fdtd_nm']:.0f} nm, 1/γ = {row['decay_plane_wave_nm']:.0f} nm")
        xx = np.linspace(0, 1, 50)
        ax.semilogy(xx, np.exp(-row["gamma_per_um"] * xx), color=c, ls="--", lw=1.0)
    ax.axhline(3e-3, color=PALETTE["muted"], ls=":", lw=1, label="≈ 3e-3: FDTD floor (truncated source line)")
    ax.set_ylim(2e-5, 1.5); ax.set_xlim(0, 1.0)
    ax.set_xlabel("x into the silica (µm)"); ax.set_ylabel("|E_z(x)| / |E_z(0)| at the beam centre")
    ax.set_title("Evanescent tails: FDTD (solid) vs plane-wave e^(−γx) (dashed);\nfits use 50 nm to 2.5 decay lengths", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax = axs[1]
    th_fine = np.linspace(THETA_C + 0.3, 60, 300)
    _, Lpw = an.gamma_evanescent(N1, N2, K0, th_fine)
    ax.plot(th_fine, 1e3 * Lpw, color=PALETTE["ink"], lw=1.5, label="plane wave: 1/γ = λ / (2π √(n1² sin²θ − n2²))")
    rows = [row for row, _ in sweep if "decay_fdtd_nm" in row]
    ax.plot([r["theta_deg"] for r in rows], [r["decay_beam_nm"] for r in rows], "s", ms=6, mfc="none", color=SERIES[1], label=f"finite beam (w0 = {fdtd.W0:g} µm), analytic spectrum, same fit window")
    ax.plot([r["theta_deg"] for r in rows], [r["decay_fdtd_nm"] for r in rows], "o", ms=6, color=SERIES[0], label="Meep FDTD, fitted on the beam centre line")
    ax.axvline(THETA_C, color=PALETTE["muted"], ls=":", label=f"θc = {THETA_C:.1f}°")
    ax.axvline(THETA_REF, color=PALETTE["green"], ls=":", label=f"strip-mode angle {THETA_REF:.1f}° (n_eff = 2.5): {1e3/ (K0*np.sqrt(REF.neff**2-N2**2)):.0f} nm")
    ax.set_ylim(0, 700); ax.set_xlim(20, 60)
    ax.set_xlabel("angle of incidence θ (deg)"); ax.set_ylabel("decay length (nm)")
    ax.set_title("Decay length vs angle: the tail lengthens without bound at θc;\nnear θc a finite beam is never purely evanescent", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "evanescent_decay.png"); plt.close(fig)


def fig_transmission(sweep):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    th_fine = np.linspace(5, 60, 400)
    beta = N1 * K0 * np.sin(np.radians(th_fine))
    _, _, Tpw = an.fresnel_s(N1, N2, K0, beta)
    Tbeam = [an.GaussianBeamSpectrum(N1, N2, K0, t, fdtd.W0, nbeta=1201).transmission_beam() for t in th_fine[::4]]
    rows = [row for row, _ in sweep]
    for ax, logy in zip(axs, (False, True)):
        ax.plot(th_fine, Tpw, color=PALETTE["ink"], lw=1.5, label="plane wave, s-pol Fresnel T = (k_x2/k_x1)|t|²")
        ax.plot(th_fine[::4], Tbeam, color=SERIES[1], ls="--", label=f"finite beam w0 = {fdtd.W0:g} µm (angular spectrum average)")
        ax.plot([r["theta_deg"] for r in rows], [r["T_fdtd"] for r in rows], "o", color=SERIES[0], label="Meep FDTD: flux into silica / incident")
        ax.axvline(THETA_C, color=PALETTE["muted"], ls=":", label=f"θc = {THETA_C:.2f}°")
        ax.set_xlabel("angle of incidence θ (deg)")
        ax.set_xlim(5, 60)
    axs[0].set_ylim(-0.02, 1.02); axs[0].set_ylabel("fraction of the incident power crossing into silica")
    axs[0].set_title("Power transmitted across the interface: a finite beam softens the\ncritical-angle edge over its angular spread ±λ/(π n1 w0)", fontsize=10)
    axs[1].set_yscale("log"); axs[1].set_ylim(1e-6, 2); axs[1].set_ylabel("same, log scale")
    axs[1].set_title("Above θc the FDTD value sits on a ~1e-4 numerical floor\n(truncation of the source line), the analytic beam value is far lower", fontsize=10)
    axs[0].legend(fontsize=7, loc="upper right")
    axs[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "transmission_vs_angle.png"); plt.close(fig)


def fig_ftir_field(r):
    gap = 0.20
    Sx, Sy = poynting(r)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1, 1]), layout="constrained")
    ax = axs[0]
    E = np.real(r["Ez"]).T
    m = 0.9 * np.abs(E[(r["x"][None, :] < -0.5) & (np.abs(r["y"][:, None] - fdtd.Y_FOCUS) < 3)]).max()
    im0 = ax.imshow(E / m, origin="lower", extent=field_extent(r), cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    fig.colorbar(im0, ax=ax, fraction=0.04, pad=0.02, label="Re E_z / (0.9 × peak of the incident beam)")
    draw_interfaces(ax, gap); ax.grid(False)
    ax.set_xlabel("x, normal to the interface (µm)"); ax.set_ylabel("y, along the interface (µm)")
    ax.set_title("Re E_z: θ = 40°, silica gap 200 nm between two silicon half-spaces;\npart of the beam tunnels and continues on the far side", fontsize=9)
    media_labels(ax, gap)
    ax = axs[1]
    x, y = r["x"], r["y"]
    mx = (x > -0.8) & (x < 1.0); my = (y > -1.0) & (y < 2.5)
    S = np.hypot(Sx, Sy)[np.ix_(mx, my)]; Smax = S.max()
    im1 = ax.imshow(S.T / Smax, origin="lower", extent=[x[mx][0], x[mx][-1], y[my][0], y[my][-1]], cmap="Blues",
                    norm=LogNorm(vmin=1e-3, vmax=1), aspect="equal")
    fig.colorbar(im1, ax=ax, fraction=0.05, pad=0.02, label="|⟨S⟩| / max (log, 3 decades)")
    step = 5
    xs, ys = x[mx][::step], y[my][::step]
    U, V = Sx[np.ix_(mx, my)][::step, ::step].T, Sy[np.ix_(mx, my)][::step, ::step].T
    mag = np.hypot(U, V); keep = mag > 1e-3 * Smax
    ax.quiver(xs, ys, np.where(keep, U / np.where(mag > 0, mag, 1), np.nan), np.where(keep, V / np.where(mag > 0, mag, 1), np.nan),
              color=PALETTE["orange"], scale=30, width=0.004, headwidth=4)
    draw_interfaces(ax, gap); ax.grid(False)
    ax.set_xlabel("x, normal to the interface (µm)"); ax.set_ylabel("y, along the interface (µm)")
    ax.set_title("⟨S⟩ direction: tilted across the gap,\na propagating beam again in the second slab", fontsize=9)
    ax = axs[2]
    _, jy = beam_column(r)
    mm = (x > -0.5) & (x < 0.9)
    E2 = np.abs(r["Ez"][:, jy]) ** 2; sc = E2[np.argmin(np.abs(x))]
    ax.semilogy(x[mm], E2[mm] / sc, color=SERIES[0], label="|E_z|² at the beam centre (FDTD)")
    Px = np.array([np.trapezoid(Sx[ix, :], y) for ix in range(len(x))]) / r["P_inc_S"]
    ax.semilogy(x[mm], np.abs(Px[mm]), color=SERIES[1], label="net normal power ∫⟨S_x⟩dy / P_inc")
    ax.axvspan(0, gap, color=PALETTE["line"], alpha=0.6, lw=0)
    g, L = an.gamma_evanescent(N1, N2, K0, 40.0)
    T = RESULTS["ftir_showcase_gap200nm_40deg"]
    ax.axhline(T["T_fdtd"], color=SERIES[1], ls=":", lw=1)
    ax.text(0.88, T["T_fdtd"] * 1.4, f"T = {T['T_fdtd']:.3f} (FDTD), {T['T_plane_wave']:.3f} (plane wave)", fontsize=8, color=SERIES[1], ha="right")
    ax.set_ylim(1e-3, 3); ax.set_xlabel("x, normal to the interface (µm), beam centre line"); ax.set_ylabel("normalised to the value at x = 0")
    ax.set_title(f"Across the gap (grey) |E_z|² falls as e^(−2γx), 1/γ = {1e3*L:.0f} nm;\nthe net normal power is constant = the tunnelled fraction T", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(OUT / "ftir_field.png"); plt.close(fig)


def fig_ftir_gap(ftir):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    gaps = np.linspace(0.03, 0.5, 200)
    for i, th in enumerate(FTIR_ANGLES):
        d = ftir[th]; c = SERIES[i]
        ax.semilogy(1e3 * gaps, an.ftir_closed_form(N1, N2, K0, th, gaps), color=c, lw=1.4,
                    label=f"θ = {th:.1f}° plane wave (closed form), 1/γ = {d['decay_length_nm']:.0f} nm")
        ax.semilogy([row["gap_nm"] for row in d["rows"]], [row["T_beam"] for row in d["rows"]], "s", mfc="none", color=c, ms=7, label=f"θ = {th:.1f}° finite-beam analytic")
        ax.semilogy([row["gap_nm"] for row in d["rows"]], [row["T_fdtd"] for row in d["rows"]], "o", color=c, ms=6,
                    label=f"θ = {th:.1f}° Meep FDTD: slope {d['fitted_slope_per_um']:.1f}/µm vs 2γ = {d['two_gamma_per_um']:.1f}/µm")
    if RESULTS["tir_floor_T_fdtd"] is not None:
        ax.axhline(RESULTS["tir_floor_T_fdtd"], color=PALETTE["muted"], ls=":", lw=1)
        ax.text(250, RESULTS["tir_floor_T_fdtd"] * 1.4, "single-interface FDTD floor", fontsize=7, color=PALETTE["muted"])
    d = ftir[THETA_REF]
    ax.text(250, 0.4, f"at the strip-mode angle {THETA_REF:.1f}°:\n+10 nm gap → tunnelled power × {1 + d['dT_over_T_per_10nm_pct_fdtd']/100:.2f} (FDTD)\n"
            f"analytic e^(−2γ·10 nm) = {1 + d['dT_over_T_per_10nm_pct_analytic']/100:.2f}", fontsize=8, color=PALETTE["ink2"])
    ax.set_ylim(1e-4, 1.2); ax.set_xlim(0, 500)
    ax.set_xlabel("silica gap between the two silicon half-spaces (nm)"); ax.set_ylabel("tunnelled power fraction T")
    ax.set_title("Frustrated TIR: the tunnelled power falls as e^(−2γ·gap), the decay length sets the slope\n(this exponential is what makes the ring-bus coupling κ² so gap-sensitive)", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left", title="closed form: T = 1/(1 + ((k1²+γ²)²/4k1²γ²) sinh²(γd))", title_fontsize=6.5)
    fig.tight_layout(); fig.savefig(OUT / "ftir_vs_gap.png"); plt.close(fig)


# =============================================================================== videos
def video(runs, labels, name, suptitle, gaps=None):
    gaps = gaps or [None] * len(runs)
    n = len(runs)
    nfr = min(len(r["frames"]) for r in runs)
    ref = runs[0]["frames"][-1].astype(np.float32)
    m = 0.8 * np.abs(ref[runs[0]["x"] < -0.5]).max()
    fig, axs = plt.subplots(1, n, figsize=(12.8, 5.4), dpi=100, sharey=True, layout="constrained")
    ims = []
    for ax, r, lab, gap in zip(axs, runs, labels, gaps):
        im = ax.imshow(r["frames"][0].astype(np.float32).T / m, origin="lower", extent=field_extent(r),
                       cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", animated=True)
        draw_interfaces(ax, gap); ax.grid(False)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("x, normal to the interface (µm)")
        media_labels(ax, gap)
        ims.append(im)
    axs[0].set_ylabel("y, along the interface (µm)")
    fig.colorbar(ims[-1], ax=axs.tolist(), fraction=0.02, pad=0.02, label="E_z(t) / (0.8 × peak in the silicon)")
    tt = fig.suptitle("", fontsize=10)

    def update(i):
        for im, r in zip(ims, runs):
            im.set_data(r["frames"][i].astype(np.float32).T / m)
        tt.set_text(f"{suptitle}\nt = {runs[0]['times'][i] * TU_FS:6.1f} fs  (E_z, red/blue = ±; the interface is the dashed line)")
        return ims + [tt]

    ani = FuncAnimation(fig, update, frames=nfr, blit=False)
    ani.save(OUT / f"{name}.mp4", writer=FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    # contact sheet
    picks = [int(p * (nfr - 1)) for p in (0.2, 0.3, 0.42, 0.6, 1.0)]
    fig, axs = plt.subplots(len(picks), n, figsize=(3.2 * n + 0.8, 2.6 * len(picks)), sharex=True, sharey=True, layout="constrained")
    for row, i in zip(axs, picks):
        for ax, r, lab, gap in zip(row, runs, labels, gaps):
            im = ax.imshow(r["frames"][i].astype(np.float32).T / m, origin="lower", extent=field_extent(r), cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            draw_interfaces(ax, gap); ax.grid(False)
            ax.set_title(f"{lab}, t = {r['times'][i] * TU_FS:.0f} fs", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("x, normal to the interface (µm)", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("y, along the interface (µm)", fontsize=8)
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02, pad=0.02, label="E_z(t) / (0.8 × peak in the silicon)")
    fig.suptitle(suptitle + "  (E_z, red/blue = ±)", fontsize=10)
    fig.savefig(OUT / f"{name}_frames.png", dpi=90); plt.close(fig)


# =============================================================================== tools.json
def write_tools():
    s = RESULTS["showcase"]; sw = RESULTS["angle_sweep"]; ft = RESULTS["ftir"]
    r40 = s[f"{TIR_SHOW:g}_deg"]
    below = [(k.replace("_deg", ""), d) for k, d in s.items() if "snell_angle_deg" in d]   # showcase angles below critical, if any
    k15, r15 = below[0] if below else (None, None)
    at_c = s.get("24.5_deg") or next((d for k, d in s.items() if abs(float(k.split("_")[0]) - THETA_C) < 0.5), None)
    below_txt = (f"{k15} deg: "
                 f"T_fdtd = {r15['T_fdtd']:.4f} vs beam-averaged Fresnel {r15['T_beam_analytic']:.4f} "
                 f"({100*(r15['T_fdtd']/r15['T_beam_analytic']-1):+.1f} %); transmitted Poynting direction {r15['transmitted_direction_from_poynting_deg']:.1f} deg "
                 f"vs Snell {r15['snell_angle_deg']:.1f} deg. ") if r15 else ""
    at_c_txt = (f"beam-averaged T at the critical angle = {at_c['T_beam_analytic']:.3f} vs FDTD {at_c['T_fdtd']:.3f}." if at_c else "")
    rc_txt = "Resolution check (flux only): " + "; ".join(
        f"{k.replace('_deg', ' deg').replace('_gap', ', gap ')} T_fdtd vs beam average " +
        ", ".join(f"{res} px/um {pct:+.1f} %" for res, pct in e["T_fdtd_vs_beam_pct_by_resolution"].items())
        for k, e in RESULTS["resolution_check"].items()) + " (monotone grid convergence). "
    tools = [
        {
            "tool": "Meep (pymeep) 2-D FDTD with GaussianBeamSource, DFT field monitors and flux monitors",
            "version": mp.__version__,
            "what_it_is": "MIT's open-source finite-difference time-domain solver: it steps Maxwell's curl equations on a Yee grid in time, "
                          "so any geometry and any source can be simulated without assuming a mode or a plane wave. Normally used for "
                          "photonic-crystal, waveguide and resonator design.",
            "used_for": "A 2-D s-polarised (E_z) Gaussian beam launched inside silicon at a chosen angle onto a flat silicon/silica "
                        "interface (and a silicon | silica gap | silicon sandwich for frustrated TIR). A narrow-band pulse with DFT "
                        "monitors gives the steady-state phasors E_z, H_x, H_y at 1310 nm; flux monitors give the power crossing "
                        "into the silica; a continuous-wave run gives the video frames. "
                        f"{RESULTS['run_counts']['total_meep_runs']} Meep runs in total: {RESULTS['run_counts']['showcase_phasor']} showcase phasor, "
                        f"{RESULTS['run_counts']['angle_sweep_phasor']} angle-sweep, {RESULTS['run_counts']['resolution_check_phasor']} flux-only resolution checks "
                        f"at {RES_CHECK} px/um, {RESULTS['run_counts']['ftir_phasor']} frustrated-TIR, {RESULTS['run_counts']['cw_video']} continuous-wave.",
            "result": below_txt + rc_txt + f"{TIR_SHOW:g} deg: decay length {r40['decay_length_fdtd_nm']:.0f} nm vs 1/gamma = "
                      f"{r40['decay_length_plane_wave_nm']:.0f} nm ({r40['decay_vs_plane_wave_pct']:+.1f} %), net normal/parallel power in silica "
                      f"{r40['net_Sx_over_net_Sy_at_x0p1um']:.1e}. FTIR at {THETA_REF:.1f} deg: tunnelling slope {ft[f'{THETA_REF:g}_deg']['fitted_slope_per_um']:.1f}/um vs 2 gamma = "
                      f"{ft[f'{THETA_REF:g}_deg']['two_gamma_per_um']:.1f}/um ({ft[f'{THETA_REF:g}_deg']['slope_agreement_pct']:+.1f} %).",
            "how_to_observe": f"cd experiments/09_tir_evanescent_fdtd && ../../.meep/bin/python run.py  (about {RESULTS['runtime_s']/60:.0f} min on this laptop; "
                              "runtime_s in out/results.json). Fields: out/fields_three_angles.png, "
                              "out/poynting_three_angles.png, out/ftir_field.png; videos out/tir_beams.mp4, out/ftir.mp4. Change SHOW_ANGLES, "
                              "SWEEP_ANGLES, FTIR_GAPS, RES_SHOW or LAM_NM (the one wavelength, passed to every Meep run) in run.py, "
                              "or W0 / SRC_X in fdtd.py, and re-run.",
        },
        {
            "tool": "numpy + scipy (analytic angular-spectrum model, Fresnel/FTIR formulas, fits)",
            "version": f"numpy {np.__version__}, scipy {scipy.__version__}",
            "what_it_is": "The array and scientific-computing libraries of Python; used here for the closed-form physics the FDTD is compared with.",
            "used_for": "analytic.py: s-polarisation Fresnel coefficients, the frustrated-TIR transfer formula through a gap, the decay "
                        "constant gamma = k0 sqrt(n1^2 sin^2 theta - n2^2), a 2-D Gaussian-beam angular spectrum (so the finite beam's "
                        "transmission and decay can be predicted, not just the plane wave's), the time-averaged Poynting vector from the phasors, "
                        "and log-linear least-squares fits of decay lengths and tunnelling slopes.",
            "result": f"theta_c = {THETA_C:.2f} deg; 1/gamma at {TIR_SHOW:g} deg = {r40['decay_length_plane_wave_nm']:.1f} nm, finite-beam prediction over the same fit window "
                      f"{r40['decay_length_beam_analytic_nm']:.1f} nm; " + at_c_txt,
            "how_to_observe": "All numbers are in out/results.json (keys showcase, angle_sweep, ftir, capstone). analytic.py is importable on its own "
                              "(../../.venv/bin/python -c 'import analytic') to play with GaussianBeamSpectrum(n1, n2, k0, theta, w0).",
        },
        {
            "tool": "matplotlib (imshow, quiver, FuncAnimation + FFMpegWriter)",
            "version": matplotlib.__version__,
            "what_it_is": "The standard Python plotting library; its animation module redraws a figure frame by frame and pipes the frames to ffmpeg.",
            "used_for": "Signed-field maps (RdBu_r centred on zero), log-scale |S| maps with direction arrows (quiver), decay/transmission/tunnelling "
                        "plots, the two videos and their contact sheets.",
            "result": "out/fields_three_angles.png, out/poynting_three_angles.png, out/poynting_profiles.png, out/evanescent_decay.png, "
                      "out/transmission_vs_angle.png, out/ftir_field.png, out/ftir_vs_gap.png, out/tir_beams.mp4 (+_frames.png), out/ftir.mp4 (+_frames.png).",
            "how_to_observe": "Open the PNGs; play the mp4s in QuickTime/VLC. In poynting_three_angles.png look at the orange arrows on the silica side: "
                              "tilted at Snell's angle for 15 deg, grazing for 24.5 deg, parallel for 40 deg.",
        },
        {
            "tool": "ffmpeg",
            "version": "9.0.1 (/opt/homebrew/bin/ffmpeg)",
            "what_it_is": "The universal command-line video encoder; matplotlib calls it to turn frame sequences into H.264 mp4 files.",
            "used_for": "Encoding out/tir_beams.mp4 and out/ftir.mp4 at 30 fps.",
            "result": "Two mp4 files of about 6 s each (180 frames at 30 fps, 0.25 Meep time units = 0.83 fs between frames).",
            "how_to_observe": "/opt/homebrew/bin/ffprobe out/tir_beams.mp4 prints the duration and frame size.",
        },
    ]
    with open(OUT / "tools.json", "w") as f:
        json.dump(tools, f, indent=2)


if __name__ == "__main__":
    main()
