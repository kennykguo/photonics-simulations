"""Part B of experiment 12: a 2-D FDTD microring side-coupled to a bus (Meep).

Run with the Meep interpreter:  ../../.meep/bin/python ring_meep.py
Geometry: ring radius 6.3 um (centre-line), 500 nm wide, n = 3.5 in n = 1.45, bus 500 nm wide,
gap GAP. 2-D, out-of-plane E (Ez), which is the 2-D stand-in for the TE slab mode (notes 13).
The ring core carries the doped modulator loss (125 dB/cm) through D_conductivity, calibrated
on a straight guide first (notes 6: alpha = 4 pi n''/lambda; notes 28: a = e^{-alpha L/2}).

Steps
  B1  straight-guide eigenmode: 2-D n_eff, n_g, evanescent decay length            -> B1_mode2d.png
  B2  loss calibration: D_conductivity that gives 125 dB/cm modal loss              -> numbers
  B3  lossless gap scan: coupling Q -> kappa^2(gap), exponential, critical gap     -> B3_gap_scan.png
  B4  lossy ring, Gaussian pulse: Harminv resonances/Q + flux transmission spectrum -> B4_spectrum.png
  B5  CW drive on one resonance: field build-up movie + energy vs time              -> B5_buildup.mp4, *_frames.png, B5_buildup_energy.png
Results -> out/B_results.json
"""
import sys, pathlib, json, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE, C0
from common.units import db_per_cm_to_alpha_per_um, alpha_per_um_to_db_per_cm
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
import numpy as np
import meep as mp
mp.verbosity(0)

T0 = time.time()
R = {}
N_SI, N_CLAD = REF.n_si, REF.n_sio2
W = REF.wg_width_um                     # 0.5 um
RADIUS = REF.radius_um                  # 6.3 um centre-line radius
L_RING = 2 * np.pi * RADIUS             # 39.58 um geometric round trip
GAP = 0.10                              # um, ring-bus gap for the main runs (see README: why so small in 2-D)
RES_MAIN = 30                           # px/um for the spectrum run
RES_CW = RES_MAIN                       # px/um for the build-up movie: MUST equal RES_MAIN (the resonance shifts by > 1 FWHM between grids)
RES_SCAN = 20                           # px/um for the gap scan
LAM0 = REF.lambda_nm * 1e-3             # 1.31 um
F0 = 1 / LAM0                           # Meep frequency units (1/um)
FW = 0.06                               # source bandwidth (1/um): covers ~ +-50 nm
CLAD = mp.Medium(index=N_CLAD)
ALPHA_TARGET = db_per_cm_to_alpha_per_um(REF.loss_db_cm_doped)   # 1/um power
R.update({"meep_version": mp.__version__, "gap_um": GAP, "res_main": RES_MAIN, "res_cw": RES_CW, "res_scan": RES_SCAN, "L_ring_geom_um": L_RING,
          "alpha_target_per_um": ALPHA_TARGET, "a_target": float(np.exp(-ALPHA_TARGET * L_RING / 2))})

def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------------------
# B1. Straight 2-D guide: n_eff, n_g, mode profile and decay length
# ---------------------------------------------------------------------------
sim = mp.Simulation(cell_size=mp.Vector3(4, 6, 0),
                    geometry=[mp.Block(size=mp.Vector3(mp.inf, W, mp.inf), material=mp.Medium(index=N_SI))],
                    default_material=CLAD, resolution=40, boundary_layers=[mp.PML(1.0)])
sim.init_sim()
where = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 4))
modes = {}
for f in (F0 * 0.98, F0, F0 * 1.02):
    m = sim.get_eigenmode(f, mp.X, where, 1, mp.Vector3(2.5, 0, 0), parity=mp.ODD_Z)
    modes[f] = (m.k.x / f, 1 / m.group_velocity)
neff2d, ng2d_direct = modes[F0]
fs = sorted(modes); dn = (modes[fs[2]][0] - modes[fs[0]][0]) / (fs[2] - fs[0])
ng2d_fd = neff2d + F0 * dn                                         # n_g = n_eff + f dn_eff/df  (notes 22)
m0 = sim.get_eigenmode(F0, mp.X, where, 1, mp.Vector3(2.5, 0, 0), parity=mp.ODD_Z)
ys = np.linspace(-2, 2, 801)
prof = np.array([m0.amplitude(mp.Vector3(0, y), mp.Ez) for y in ys])
prof = np.abs(prof) / np.abs(prof).max()
gamma_an = 2 * np.pi / LAM0 * np.sqrt(neff2d ** 2 - N_CLAD ** 2)   # 1/um (notes 16, 24)
sel = (ys > W / 2 + 0.05) & (ys < W / 2 + 0.5)
gamma_fit = -np.polyfit(ys[sel], np.log(prof[sel]), 1)[0]
# second mode check (is the 2-D 500 nm guide multimode for Ez?)
m2 = sim.get_eigenmode(F0, mp.X, where, 2, mp.Vector3(2.5, 0, 0), parity=mp.ODD_Z)
neff2d_mode2 = m2.k.x / F0
sim.reset_meep()
R.update({"neff_2d": neff2d, "ng_2d": ng2d_direct, "ng_2d_finite_difference": ng2d_fd,
          "neff_2d_mode2": neff2d_mode2, "gamma_2d_per_um": gamma_an, "decay_len_2d_nm": 1e3 / gamma_an,
          "decay_len_2d_fit_nm": 1e3 / gamma_fit,
          "gamma_3d_textbook_per_um": 2 * np.pi / LAM0 * np.sqrt(REF.neff ** 2 - N_CLAD ** 2),
          "decay_len_3d_textbook_nm": 1e3 / (2 * np.pi / LAM0 * np.sqrt(REF.neff ** 2 - N_CLAD ** 2)),
          "fsr_nm_pred_straight_ng": LAM0 ** 2 / (ng2d_direct * L_RING) * 1e3,
          "fsr_thz_pred_straight_ng": C0 / (ng2d_direct * L_RING * 1e-6) / 1e12})
log(f"B1 2-D guide: n_eff = {neff2d:.4f} (2nd mode {neff2d_mode2:.3f}: multimode!), n_g = {ng2d_direct:.4f} (finite diff {ng2d_fd:.4f}); "
    f"1/γ = {1e3/gamma_an:.0f} nm (fit {1e3/gamma_fit:.0f} nm) vs textbook 3-D {R['decay_len_3d_textbook_nm']:.0f} nm")
log(f"B1 predicted FSR with straight n_g and L = 2πR: {R['fsr_nm_pred_straight_ng']:.2f} nm = {R['fsr_thz_pred_straight_ng']:.3f} THz")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(ys * 1e3, prof, color=SERIES[0], label=f"|E_z(y)| of the fundamental 2-D mode, n_eff = {neff2d:.3f}")
ax.plot(ys[ys > W / 2] * 1e3, prof[np.argmin(np.abs(ys - W / 2))] * np.exp(-gamma_an * (ys[ys > W / 2] - W / 2)), "--",
        color=SERIES[1], label=f"e^{{−γ(y−w/2)}}, 1/γ = λ/(2π√(n_eff²−n₂²)) = {1e3/gamma_an:.0f} nm")
ax.axvspan(-W / 2 * 1e3, W / 2 * 1e3, color=SERIES[0], alpha=0.08)
ax.set(xlabel="transverse position y (nm)", ylabel="|E_z| (normalised)", xlim=(-1200, 1200),
       title=f"2-D 500 nm guide (n = 3.5) at 1310 nm: n_eff {neff2d:.3f}, n_g {ng2d_direct:.3f}\nthe evanescent tail is only {1e3/gamma_an:.0f} nm long (3-D strip: ~100 nm)")
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "B1_mode2d.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# B2. Loss calibration: which D_conductivity gives 125 dB/cm *modal* loss?
# ---------------------------------------------------------------------------
def straight_loss(sigma):
    lossy = mp.Medium(index=N_SI, D_conductivity=sigma)
    sim = mp.Simulation(cell_size=mp.Vector3(14, 6, 0),
                        geometry=[mp.Block(size=mp.Vector3(mp.inf, W, mp.inf), material=lossy)],
                        sources=[mp.EigenModeSource(mp.GaussianSource(F0, fwidth=0.05), center=mp.Vector3(-5.5, 0),
                                                    size=mp.Vector3(0, 4), eig_band=1, eig_parity=mp.ODD_Z)],
                        default_material=CLAD, resolution=40, boundary_layers=[mp.PML(1.0)])
    fa = sim.add_flux(F0, 0, 1, mp.FluxRegion(center=mp.Vector3(-4, 0), size=mp.Vector3(0, 4)))
    fb = sim.add_flux(F0, 0, 1, mp.FluxRegion(center=mp.Vector3(4, 0), size=mp.Vector3(0, 4)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, mp.Vector3(4, 0), 1e-6))
    Pa, Pb = mp.get_fluxes(fa)[0], mp.get_fluxes(fb)[0]
    sim.reset_meep()
    return -np.log(Pb / Pa) / 8.0                                  # power alpha over 8 um, 1/um

sigma0 = ALPHA_TARGET / N_SI                                       # bulk plane-wave estimate: alpha = n sigma_D
alpha0 = straight_loss(sigma0)
SIGMA = sigma0 * ALPHA_TARGET / alpha0                             # linear correction (loss is linear in sigma here)
alpha1 = straight_loss(SIGMA)
R.update({"sigma_D_bulk_guess": sigma0, "alpha_meas_at_guess_per_um": alpha0, "db_cm_at_guess": alpha_per_um_to_db_per_cm(alpha0),
          "sigma_D_calibrated": SIGMA, "alpha_meas_calibrated_per_um": alpha1, "db_cm_calibrated": alpha_per_um_to_db_per_cm(alpha1),
          "a_from_calibrated_loss": float(np.exp(-alpha1 * L_RING / 2)),
          "n_pp_2d": alpha1 * LAM0 / (4 * np.pi),
          "q_intrinsic_pred": 2 * np.pi * ng2d_direct / (alpha1 * LAM0)})
log(f"B2 loss: σ_D = α/n = {sigma0:.3e} gives {alpha_per_um_to_db_per_cm(alpha0):.1f} dB/cm; calibrated σ_D = {SIGMA:.3e} gives "
    f"{alpha_per_um_to_db_per_cm(alpha1):.1f} dB/cm (target {REF.loss_db_cm_doped}); a = e^(−αL/2) = {R['a_from_calibrated_loss']:.4f}; "
    f"predicted intrinsic Q = 2π n_g/(α λ) = {R['q_intrinsic_pred']:.0f}")

# ---------------------------------------------------------------------------
# shared ring geometry
# ---------------------------------------------------------------------------
CELL = mp.Vector3(17, 18, 0)
YC = 0.8                                                 # ring centre y
def bus_y(gap): return YC - (RADIUS + W + gap)           # bus centre-line y
def ring_sim(gap, sigma, res, source):
    core = mp.Medium(index=N_SI, D_conductivity=sigma)
    yb = bus_y(gap)
    geom = [mp.Cylinder(radius=RADIUS + W / 2, center=mp.Vector3(0, YC), material=core),
            mp.Cylinder(radius=RADIUS - W / 2, center=mp.Vector3(0, YC), material=CLAD),
            mp.Block(size=mp.Vector3(mp.inf, W, mp.inf), center=mp.Vector3(0, yb), material=mp.Medium(index=N_SI))]
    src = [mp.EigenModeSource(source, center=mp.Vector3(-7.0, yb), size=mp.Vector3(0, 2), eig_band=1, eig_parity=mp.ODD_Z)]
    return mp.Simulation(cell_size=CELL, geometry=geom, sources=src, default_material=CLAD, resolution=res,
                         boundary_layers=[mp.PML(1.0)])
RING_PT = mp.Vector3(RADIUS * np.cos(0.7), YC + RADIUS * np.sin(0.7))   # Harminv probe on the ring centre-line

def fundamental_family(hmodes, amp_frac=0.25, qmin=300):
    """Keep Harminv modes of the fundamental ring mode: strong amplitude, sensible Q, regular spacing."""
    ms = [(m.freq, m.Q, abs(m.amp)) for m in hmodes if m.Q > qmin and m.freq > 0]
    if not ms: return []
    amax = max(a for _, _, a in ms)
    ms = sorted([m for m in ms if m[2] > amp_frac * amax])
    out = []                                             # merge duplicates closer than 0.5 nm (keep the stronger)
    for m in ms:
        if out and abs(1 / m[0] - 1 / out[-1][0]) < 0.0005:
            if m[2] > out[-1][2]: out[-1] = m
        else:
            out.append(m)
    return out

def t_from_Qc(Qc, ng, L, lam):
    """Lossless ring (a = 1): Q = π n_g L √t / (λ (1 − t))  ->  solve for t."""
    qp = Qc * lam / (np.pi * ng * L)
    x = (-1 + np.sqrt(1 + 4 * qp ** 2)) / (2 * qp)                # x = √t
    return x ** 2

def ta_from_Q(Q, ng, L, lam):
    """Loaded Q = π n_g L √(ta) / (λ (1 − ta))  ->  ta."""
    return t_from_Qc(Q, ng, L, lam)

# ---------------------------------------------------------------------------
# B3. Gap scan: kappa^2 measured DIRECTLY from the first pass of the pulse through the coupler
# ---------------------------------------------------------------------------
# The Gaussian pulse (~1/fwidth = 17 time units long) is much shorter than one round trip
# (n_g L = 142 units), so a flux monitor across the ring a quarter lap after the coupler sees the
# first-pass pulse alone if we read it at T_FIRST, before the second pass arrives.  The ratio to the
# bus input flux is the power coupling kappa^2(lambda) of the point coupler (notes 27: K^2 = sin^2).
T_FIRST = 220.0
RING_FLUX = mp.FluxRegion(center=mp.Vector3(RADIUS, YC), size=mp.Vector3(1.2, 0))     # crosses the ring at theta = 0, normal +y (CCW travel)
def add_monitors(sim, gap, df=0.04, nf=4000):
    yb = bus_y(gap)
    fin = sim.add_flux(F0, df, nf, mp.FluxRegion(center=mp.Vector3(-6.3, yb), size=mp.Vector3(0, 2)))
    fring = sim.add_flux(F0, df, nf, RING_FLUX)
    fout = sim.add_flux(F0, df, nf, mp.FluxRegion(center=mp.Vector3(6.8, yb), size=mp.Vector3(0, 2)))
    return fin, fring, fout

def kappa2_at(freqs, k2, lam_nm):
    return float(np.interp(1e3 / lam_nm, freqs, k2))

gaps = [0.10, 0.15, 0.20]
scan = []
for g in gaps:
    sim = ring_sim(g, 0.0, RES_SCAN, mp.GaussianSource(F0, fwidth=FW))
    fin, fring, fout = add_monitors(sim, g, nf=400)
    sim.run(until=T_FIRST)
    fr = np.array(mp.get_flux_freqs(fring)); Pin = np.array(mp.get_fluxes(fin)); Pring1 = np.array(mp.get_fluxes(fring))
    k2 = Pring1 / Pin
    h = mp.Harminv(mp.Ez, RING_PT, F0, 0.05)
    sim.run(h, until=1200)                   # 1200 more time units for Harminv
    fam = fundamental_family(h.modes)
    sim.reset_meep()
    near = [m for m in fam if abs(1 / m[0] - LAM0) < 0.02]
    Qc = float(np.median([m[1] for m in near])) if near else float("nan")
    t_h = t_from_Qc(Qc, ng2d_direct, L_RING, LAM0)
    k2_1310 = kappa2_at(fr, k2, REF.lambda_nm)
    scan.append({"gap_um": g, "kappa2_first_pass_1310": k2_1310, "t_first_pass": float(np.sqrt(1 - k2_1310)),
                 "kappa2_vs_lambda": {"lambda_nm": (1e3 / fr).tolist(), "kappa2": k2.tolist()},
                 "Q_c_harminv": Qc, "kappa2_from_harminv_Q": 1 - t_h ** 2,
                 "resonances_nm": [1e3 / m[0] for m in fam], "Qs": [m[1] for m in fam]})
    log(f"B3 gap {g:.2f} µm (lossless): first-pass κ²(1310) = {k2_1310:.4f} (t = {np.sqrt(1-k2_1310):.4f}); Harminv Q_c ≈ {Qc:.0f} -> κ² = 1−t² = {1-t_h**2:.4f}; "
        f"resonances {np.round([1e3/m[0] for m in fam],1)}")
g_arr = np.array([s["gap_um"] for s in scan]); k2_arr = np.array([s["kappa2_first_pass_1310"] for s in scan])
slope, icpt = np.polyfit(g_arr, np.log(k2_arr), 1)
gap_crit = (np.log(1 - R["a_target"] ** 2) - icpt) / slope
R.update({"gap_scan": scan, "kappa2_decay_per_um_fit": -slope, "kappa2_decay_per_um_expected_2gamma": 2 * gamma_an,
          "gap_for_critical_coupling_um": gap_crit, "kappa2_needed_for_critical": 1 - R["a_target"] ** 2,
          "kappa2_at_zero_gap_extrapolated": float(np.exp(icpt))})
log(f"B3 ln κ² slope = {slope:.1f} /µm (expected ≈ −2γ = {-2*gamma_an:.1f} /µm); critical coupling (κ² = 1−a² = {1-R['a_target']**2:.3f}) "
    f"extrapolates to gap ≈ {gap_crit*1e3:.0f} nm (κ² at zero gap would be {np.exp(icpt):.3f})")
fig, ax = plt.subplots(figsize=(6.8, 4))
gg = np.linspace(0.0, 0.25, 50)
ax.semilogy(g_arr * 1e3, k2_arr, "o", color=SERIES[0], ms=7, label="κ² at 1310 nm from the first-pass flux (direct)")
ax.semilogy(g_arr * 1e3, [s["kappa2_from_harminv_Q"] for s in scan], "s", color=SERIES[2], ms=6, label="κ² inferred from the Harminv coupling Q (less reliable, Q > 10⁴)")
ax.semilogy(gg * 1e3, np.exp(icpt + slope * gg), "-", color=SERIES[0], lw=1.2, label=f"fit κ² ∝ e^{{−{-slope:.1f}·gap}} (2γ of the tail = {2*gamma_an:.1f} /µm)")
ax.axhline(1 - R["a_target"] ** 2, color=SERIES[1], ls="--", label=f"critical coupling for a = {R['a_target']:.3f}: κ² = 1 − a² = {1-R['a_target']**2:.3f}")
ax.axhline(REF.kappa2, color=SERIES[3], ls=":", label=f"reference 3-D device: κ² = {REF.kappa2} at a 200 nm gap")
ax.set(xlabel="ring–bus gap (nm)", ylabel="power coupling κ² per pass", xlim=(0, 250),
       title="κ² falls exponentially with the gap; the doped ring\nreaches critical coupling only at a ~40 nm gap in this 2-D model")
ax.legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "B3_gap_scan.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# B4. Lossy ring: pulse -> Harminv + transmission spectrum
# ---------------------------------------------------------------------------
def run_spectrum(with_ring):
    yb = bus_y(GAP)
    if with_ring:
        sim = ring_sim(GAP, SIGMA, RES_MAIN, mp.GaussianSource(F0, fwidth=FW))
    else:
        sim = mp.Simulation(cell_size=CELL, geometry=[mp.Block(size=mp.Vector3(mp.inf, W, mp.inf), center=mp.Vector3(0, yb), material=mp.Medium(index=N_SI))],
                            sources=[mp.EigenModeSource(mp.GaussianSource(F0, fwidth=FW), center=mp.Vector3(-7.0, yb), size=mp.Vector3(0, 2), eig_band=1, eig_parity=mp.ODD_Z)],
                            default_material=CLAD, resolution=RES_MAIN, boundary_layers=[mp.PML(1.0)])
    fin, fring, fout = add_monitors(sim, GAP)
    k2 = None
    if with_ring:
        sim.run(until=T_FIRST)                                   # first pass through the coupler only
        k2 = np.array(mp.get_fluxes(fring)) / np.array(mp.get_fluxes(fin))
        h = mp.Harminv(mp.Ez, RING_PT, F0, 0.05)
        decayed = mp.stop_when_fields_decayed(100, mp.Ez, RING_PT, 3e-4)    # probe INSIDE the ring (|E|^2 down by 3e-4 ~ 4 energy lifetimes)
        def stop(sim):  # decay criterion with a hard cap (long-lived higher-order modes otherwise run forever)
            return decayed(sim) or sim.meep_time() > 7000
        sim.run(h, until=stop)
    else:
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(6.8, yb), 1e-6))
    freqs = np.array(mp.get_flux_freqs(fout)); Pout = np.array(mp.get_fluxes(fout)); Pin = np.array(mp.get_fluxes(fin))
    tend = sim.meep_time()
    hm = fundamental_family(h.modes) if with_ring else []
    hall = [(m.freq, m.Q, abs(m.amp)) for m in h.modes] if with_ring else []
    sim.reset_meep()
    return freqs, Pout, Pin, hm, hall, tend, k2

freqs, P0, Pin0, _, _, t_norm, _ = run_spectrum(False)
log(f"B4 normalisation run done (t_end = {t_norm:.0f})")
freqs, P1, Pin1, hm, hall, t_ring, k2_main = run_spectrum(True)
log(f"B4 ring run done (t_end = {t_ring:.0f} µm/c = {t_ring*LAM0/LAM0:.0f} time units, ≈ {t_ring/LAM0:.0f} optical periods)")
freqs_k2 = freqs.copy()
lam = 1e3 / freqs                                     # nm
T = P1 / P0
order = np.argsort(lam); lam, T = lam[order], T[order]
band = (lam > 1280) & (lam < 1345)                    # keep the well-excited part of the source band
lam, T = lam[band], T[band]

def dips_of(lam, T, depth=0.15):
    i = np.where((T[1:-1] < T[:-2]) & (T[1:-1] <= T[2:]) & (T[1:-1] < 1 - depth))[0] + 1
    return i

idx = dips_of(lam, T)
harm_lam = np.array([1e3 / m[0] for m in hm]); harm_Q = np.array([m[1] for m in hm])
# match flux dips to the Harminv fundamental family (within 1.5 nm)
matched = [i for i in idx if np.min(np.abs(harm_lam - lam[i])) < 1.5] if len(harm_lam) else list(idx)
lam_dips = lam[matched]; T_dips = T[matched]
def fwhm_dip(lam, T, i, interpolate=True):
    """Full width of the notch at half depth. The flux grid is ~17 pm (5 % of the notch), so the two half-depth
    crossings are linearly interpolated between the grid points; interpolate=False returns the raw grid width
    (first grid point at or above half depth on each side), which overestimates the width by 0 to 2 grid steps."""
    half = (1 + T[i]) / 2
    lo = i
    while lo > 0 and T[lo] < half: lo -= 1
    hi = i
    while hi < len(T) - 1 and T[hi] < half: hi += 1
    if not interpolate or T[lo] < half or T[hi] < half:
        return abs(lam[hi] - lam[lo])
    def crossing(j_out, j_in):          # T[j_out] >= half > T[j_in]: linear interpolation of lambda at T = half
        return lam[j_in] + (half - T[j_in]) * (lam[j_out] - lam[j_in]) / (T[j_out] - T[j_in])
    return abs(crossing(hi, hi - 1) - crossing(lo, lo + 1))
fw = np.array([fwhm_dip(lam, T, i) for i in matched])
fw_grid = np.array([fwhm_dip(lam, T, i, interpolate=False) for i in matched])
flux_grid_step_pm = float(np.median(np.diff(lam[np.abs(lam - REF.lambda_nm) < 5])) * 1e3)
Q_flux = lam_dips / fw
i_c = int(np.argmin(np.abs(lam_dips - REF.lambda_nm)))
fsr_fdtd = float(np.mean(np.diff(lam_dips))) if len(lam_dips) > 1 else float("nan")
harm_near = harm_lam[np.argmin(np.abs(harm_lam - lam_dips[i_c]))]
Q_harm_c = harm_Q[np.argmin(np.abs(harm_lam - lam_dips[i_c]))]
fsr_harm = float(np.mean(np.diff(np.sort(harm_lam[np.abs(harm_lam - REF.lambda_nm) < 40])))) if len(harm_lam) > 1 else float("nan")
# invert the all-pass formulas on the central dip: loaded Q -> ta ; T_min -> (t-a) (under-coupled branch, t > a)
ta = ta_from_Q(Q_flux[i_c], ng2d_direct, L_RING, lam_dips[i_c] * 1e-3)
r = np.sqrt(T_dips[i_c]) * (1 - ta)                 # t - a
# t - a = r, t a = ta  ->  t = (r + sqrt(r^2 + 4 ta))/2
t_fd = (r + np.sqrt(r ** 2 + 4 * ta)) / 2; a_fd = ta / t_fd
k2_direct = kappa2_at(freqs_k2, k2_main, lam_dips[i_c])          # first-pass kappa^2 at the central resonance (lossy main run)
t_scan = float(np.sqrt(1 - k2_direct))
Tmin_pred = ((t_scan - R["a_from_calibrated_loss"]) / (1 - t_scan * R["a_from_calibrated_loss"])) ** 2
Q_pred = np.pi * ng2d_direct * L_RING * np.sqrt(t_scan * R["a_from_calibrated_loss"]) / (LAM0 * (1 - t_scan * R["a_from_calibrated_loss"]))
fwhm_pred_pm = lam_dips[i_c] * 1e3 / Q_pred
R.update({"t_end_ring_run": t_ring, "t_end_norm_run": t_norm,
          "harminv_all": [{"lambda_nm": 1e3 / f, "Q": q, "amp": a} for f, q, a in hall],
          "harminv_fundamental_nm": harm_lam.tolist(), "harminv_fundamental_Q": harm_Q.tolist(),
          "flux_dips_nm": lam_dips.tolist(), "flux_dips_Tmin": T_dips.tolist(), "flux_dips_fwhm_pm": (fw * 1e3).tolist(),
          "flux_dips_Q": Q_flux.tolist(),
          "central_dip_nm": float(lam_dips[i_c]), "central_dip_Tmin": float(T_dips[i_c]), "central_dip_Tmin_dB": float(10 * np.log10(T_dips[i_c])),
          "central_dip_fwhm_pm": float(fw[i_c] * 1e3), "central_dip_Q_flux": float(Q_flux[i_c]),
          "central_dip_fwhm_grid_pm": float(fw_grid[i_c] * 1e3), "central_dip_Q_flux_grid": float(lam_dips[i_c] / fw_grid[i_c]),
          "flux_grid_step_pm": flux_grid_step_pm, "record_in_amplitude_lifetimes": float(t_ring / (Q_harm_c / (np.pi / (lam_dips[i_c] * 1e-3)))),
          "spectrum_lambda_nm": lam.tolist(), "spectrum_T": T.tolist(),
          "central_dip_Q_harminv": float(Q_harm_c), "central_dip_lambda_harminv_nm": float(harm_near),
          "fsr_nm_fdtd_flux": fsr_fdtd, "fsr_nm_fdtd_harminv": fsr_harm,
          "fsr_thz_fdtd_flux": C0 * fsr_fdtd * 1e-9 / (lam_dips[i_c] * 1e-9) ** 2 / 1e12,
          "ngL_eff_um_from_fsr": (lam_dips[i_c] * 1e-3) ** 2 / (fsr_fdtd * 1e-3),
          "ng_bend_eff_if_L_geom": (lam_dips[i_c] * 1e-3) ** 2 / (fsr_fdtd * 1e-3) / L_RING,
          "ta_from_Q": float(ta), "t_from_fdtd": float(t_fd), "a_from_fdtd": float(a_fd),
          "kappa2_first_pass_main": float(k2_direct), "t_from_first_pass_main": float(t_scan),
          "kappa2_first_pass_main_vs_lambda": {"lambda_nm": (1e3 / freqs_k2).tolist(), "kappa2": k2_main.tolist()},
          "Tmin_pred_from_scan_and_loss": float(Tmin_pred),
          "Q_pred_from_scan_and_loss": float(Q_pred), "fwhm_pm_pred_from_scan_and_loss": float(fwhm_pred_pm),
          "kappa2_fdtd": float(1 - t_fd ** 2)})
log(f"B4 flux dips (nm): {np.round(lam_dips, 2)}; T_min: {np.round(T_dips, 3)}; FWHM (pm): {np.round(fw*1e3, 0)}; Q: {np.round(Q_flux, 0)}")
log(f"B4 Harminv fundamental family (nm / Q): {[(round(l,2), round(q)) for l, q in zip(harm_lam, harm_Q)]}")
log(f"B4 FSR: flux {fsr_fdtd:.2f} nm, Harminv {fsr_harm:.2f} nm; predicted c/(n_g L) with straight n_g {ng2d_direct:.3f}, L = 2πR: {R['fsr_nm_pred_straight_ng']:.2f} nm "
    f"-> n_g L_eff from FSR = {R['ngL_eff_um_from_fsr']:.1f} µm (geometric n_g L = {ng2d_direct*L_RING:.1f} µm)")
log(f"B4 central dip {lam_dips[i_c]:.2f} nm: T_min = {T_dips[i_c]:.3f} ({10*np.log10(T_dips[i_c]):.1f} dB), FWHM = {fw[i_c]*1e3:.0f} pm (interpolated; raw grid width {fw_grid[i_c]*1e3:.0f} pm on a {flux_grid_step_pm:.1f} pm grid), "
    f"Q_flux = {Q_flux[i_c]:.0f} (grid {lam_dips[i_c]/fw_grid[i_c]:.0f}), Q_harminv = {Q_harm_c:.0f}; record = {t_ring:.0f} µm/c = {R['record_in_amplitude_lifetimes']:.1f} amplitude lifetimes")
log(f"B4 inversion: ta = {ta:.4f} -> t = {t_fd:.4f} (κ² = {1-t_fd**2:.3f}), a = {a_fd:.4f}  [calibrated a = {R['a_from_calibrated_loss']:.4f}, first-pass κ² = {k2_direct:.4f} -> t = {t_scan:.4f}]")
log(f"B4 forward prediction from first-pass t and calibrated a: T_min = {Tmin_pred:.3f}, Q = {Q_pred:.0f}, FWHM = {fwhm_pred_pm:.0f} pm")

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
ax0 = fig.add_subplot(gs[0, :]); ax1 = fig.add_subplot(gs[1, 0]); ax2 = fig.add_subplot(gs[1, 1])
ax0.plot(lam, T, color=SERIES[0], label="FDTD through transmission P_out(ring) / P_out(straight bus)")
for l, q in zip(harm_lam, harm_Q):
    ax0.axvline(l, color=SERIES[1], lw=0.9, ls="--")
    ax0.text(l, 1.03, f"Q {q:.0f}", ha="center", fontsize=7.5, color=SERIES[1], rotation=90, va="bottom")
ax0.plot([], [], "--", color=SERIES[1], label="Harminv resonances of the fundamental ring mode (label = Q)")
other = [m for m in hall if (m[1] > 300) and (np.min(np.abs(harm_lam - 1e3 / m[0])) > 0.5 if len(harm_lam) else True)]
for f, q, a in other:
    ax0.plot(1e3 / f, 1.0, "v", color=SERIES[2], ms=5)
ax0.plot([], [], "v", color=SERIES[2], label="other (higher-order / weak) Harminv modes")
if len(lam_dips) > 1:
    ax0.annotate("", xy=(lam_dips[i_c], 0.35), xytext=(lam_dips[i_c - 1] if i_c > 0 else lam_dips[i_c + 1], 0.35), arrowprops=dict(arrowstyle="<->", color=PALETTE["ink2"]))
    ax0.text((lam_dips[i_c] + (lam_dips[i_c - 1] if i_c > 0 else lam_dips[i_c + 1])) / 2, 0.38, f"FSR = {fsr_fdtd:.2f} nm", ha="center", fontsize=9)
ax0.set(xlabel="wavelength (nm)", ylabel="through transmission T", ylim=(0, 1.3), xlim=(lam.min(), lam.max()),
        title=f"2-D FDTD ring (R = {RADIUS} µm, gap {GAP*1e3:.0f} nm, {REF.loss_db_cm_doped:.0f} dB/cm): a notch every FSR = {fsr_fdtd:.2f} nm\n"
              f"(λ²/(n_g L) with the straight-guide n_g {ng2d_direct:.3f} and L = 2πR predicts {R['fsr_nm_pred_straight_ng']:.2f} nm)")
ax0.legend(fontsize=8, loc="lower left")
zoom = np.abs(lam - lam_dips[i_c]) < 1.5
dl = np.linspace(-1.5, 1.5, 1201)
Tfit = np.abs((t_fd - a_fd * np.exp(-1j * 2 * np.pi * ng2d_direct * L_RING * (-dl * 1e-3) / (lam_dips[i_c] * 1e-3) ** 2)) /
              (1 - t_fd * a_fd * np.exp(-1j * 2 * np.pi * ng2d_direct * L_RING * (-dl * 1e-3) / (lam_dips[i_c] * 1e-3) ** 2))) ** 2
ax1.plot(lam[zoom] - lam_dips[i_c], T[zoom], color=SERIES[0], label="FDTD")
ax1.plot(dl, Tfit, "--", color=SERIES[4], lw=1.2, label=f"all-pass formula with t = {t_fd:.3f}, a = {a_fd:.3f} (from Q and T_min)")
half = (1 + T_dips[i_c]) / 2
ax1.annotate("", xy=(fw[i_c] / 2, half), xytext=(-fw[i_c] / 2, half), arrowprops=dict(arrowstyle="<->", color=SERIES[3]))
ax1.text(0, half + 0.05, f"FWHM {fw[i_c]*1e3:.0f} pm → Q = {Q_flux[i_c]:.0f}\n(Harminv: Q = {Q_harm_c:.0f})", ha="center", fontsize=8.5, color=SERIES[3])
ax1.set(xlabel=f"δ = λ − {lam_dips[i_c]:.2f} nm (nm)", ylabel="T", ylim=(0, 1.05), title=f"Central notch: T_min = {T_dips[i_c]:.2f} ({10*np.log10(T_dips[i_c]):.1f} dB),\nunder-coupled (t > a)")
ax1.legend(fontsize=8, loc="lower right")
# right: what the same ring would do at the reference (3-D) coupling and for the two regimes
for lab, t_, c in [(f"FDTD 2-D ring: t = {t_fd:.3f} (κ² = {1-t_fd**2:.3f}), under-coupled", t_fd, SERIES[0]),
                   (f"same a, t = a = {a_fd:.3f}: critical (the reference device)", a_fd, SERIES[1]),
                   ("same a, t = 0.90: over-coupled", 0.90, SERIES[2])]:
    Tt = np.abs((t_ - a_fd * np.exp(-1j * 2 * np.pi * ng2d_direct * L_RING * (-dl * 1e-3) / LAM0 ** 2)) /
                (1 - t_ * a_fd * np.exp(-1j * 2 * np.pi * ng2d_direct * L_RING * (-dl * 1e-3) / LAM0 ** 2))) ** 2
    ax2.plot(dl, Tt, color=c, label=lab)
ax2.set(xlabel="δ (nm)", ylabel="T", ylim=(0, 1.05), title="Same a, three couplers: the 100 nm gap gives\ntoo little κ² for critical coupling")
ax2.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "B4_spectrum.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# B5. CW drive on the central resonance: build-up movie and energy vs time
# ---------------------------------------------------------------------------
f_res = 1e3 / harm_near if np.isfinite(harm_near) else 1e3 / lam_dips[i_c]
f_res = 1 / (harm_near * 1e-3)
sim = ring_sim(GAP, SIGMA, RES_CW, mp.ContinuousSource(frequency=f_res, width=10))
yb = bus_y(GAP)
t_fine = np.arange(0, 12.0, 0.5)                       # 24 frames: watch the wave enter (0.8 s of video); step > T/4 (see below)
t_coarse = np.linspace(12.0, 4000.0, 276)              # 276 frames: watch the ring charge up (9.2 s); 4000 µm/c ≈ 2.2 amplitude time constants
times = np.concatenate([t_fine, t_coarse])
T_OPT = 1 / f_res                                      # optical period in Meep time units (µm/c)
frames, energies, tlist, env_last = [], [], [], None
size = mp.Vector3(CELL.x - 2, CELL.y - 2)               # skip the PML
xs = np.linspace(-size.x / 2, size.x / 2, int(size.x * RES_CW) + 1); ys2 = np.linspace(-size.y / 2, size.y / 2, int(size.y * RES_CW) + 1)
XX, YY = np.meshgrid(xs, ys2, indexing="ij")
rr = np.hypot(XX, YY - YC)
ring_mask = (rr > RADIUS - W / 2) & (rr < RADIUS + W / 2)
bus_mask = (np.abs(YY - yb) < W / 4) & (XX > -6.0) & (XX < -3.0)     # bus upstream of the coupler (same ±w/4 band as the ring)
sim.init_sim()
t_prev = 0.0
for tt in times:
    assert tt - t_prev >= 0, "frame times must be spaced by more than T/4"
    sim.run(until=tt - t_prev)                 # Meep's `until` is a DURATION from the current time
    t_prev = tt
    ez = np.real(sim.get_array(center=mp.Vector3(0, 0), size=size, component=mp.Ez))
    if ez.shape != XX.shape:  # guard against off-by-one in Meep's array sizing
        ez = ez[:XX.shape[0], :XX.shape[1]]
    frames.append(ez.astype(np.float32)); tlist.append(tt)
    # a second snapshot a quarter optical period later: Ez² + Ez(t+T/4)² is the envelope² of a narrow-band field,
    # so the stored energy does not carry the 2ω ripple that a single instantaneous snapshot would (aliased at the frame rate)
    sim.run(until=T_OPT / 4); t_prev += T_OPT / 4
    ez_q = np.real(sim.get_array(center=mp.Vector3(0, 0), size=size, component=mp.Ez))[:XX.shape[0], :XX.shape[1]]
    env2 = ez ** 2 + ez_q ** 2
    energies.append(float(np.mean(env2[ring_mask])))
    env_last = env2
ez_final = frames[-1].astype(float)
env_final = env_last
# radial position of the bent mode (where |Ez|^2 peaks, averaged over the half of the ring away from the coupler)
ang = np.arctan2(YY - YC, XX)
sect = (ang > 0.3) & (ang < np.pi - 0.3)
rbins = np.arange(RADIUS - 0.6, RADIUS + 0.6, 1 / RES_CW)
rprof = np.array([np.mean(env_final[sect & (rr >= r0) & (rr < r0 + 1 / RES_CW)]) for r0 in rbins])
r_peak = rbins[np.argmax(rprof)] + 0.5 / RES_CW
r_mean = float(np.sum((rbins + 0.5 / RES_CW) * rprof) / np.sum(rprof))
L_eff = 2 * np.pi * r_mean
# field enhancement: mean |Ez|^2 on the ring centre band vs in the bus upstream, in steady state
ring_band = sect & (np.abs(rr - r_mean) < W / 4)
enh = float(np.mean(env_final[ring_band]) / np.mean(env_final[bus_mask]))
enh_pred = (1 - t_fd ** 2) / (1 - t_fd * a_fd) ** 2                      # |E_ring/E_in|² on resonance from the inverted t, a
enh_pred_cal = (1 - t_scan ** 2) / (1 - t_scan * R["a_from_calibrated_loss"]) ** 2   # same with the first-pass t and calibrated a
# the ring is only ~87 % charged (amplitude) at t = 4000: scale the prediction to the same instant
enh_now_factor = (1 - np.exp(-tlist[-1] / (Q_harm_c / (np.pi * f_res)))) ** 2
# build-up fit: energy ∝ (1 − e^{−t/τ_a})², τ_a = 2Q/ω = Q λ/π (in µm/c units)
E = np.array(energies); tl = np.array(tlist)
sel = tl > 30
from scipy.optimize import curve_fit
def model(t, U, tau): return U * (1 - np.exp(-t / tau)) ** 2
popt, _ = curve_fit(model, tl[sel], E[sel], p0=[E[-1] * 1.3, 1500.0], bounds=([0, 10.0], [np.inf, 1e5]))
U_fit, tau_fit = popt; Q_buildup = tau_fit * np.pi * f_res      # Q = τ_a ω / 2 = τ_a π f
tau_pred = Q_harm_c / (np.pi * f_res)
E_ss = U_fit                                                     # fitted steady-state energy (the run stops before full charge)
sim.reset_meep()
R.update({"cw_frequency": f_res, "cw_lambda_nm": 1e3 / f_res, "r_peak_um": float(r_peak), "r_mean_um": r_mean, "L_eff_um": L_eff,
          "fsr_nm_pred_bend_corrected": LAM0 ** 2 / (ng2d_direct * L_eff) * 1e3,
          "buildup_tau_amp_fit": float(tau_fit), "buildup_tau_amp_pred_from_harminv_Q": float(tau_pred),
          "Q_from_buildup": float(Q_buildup), "buildup_tau_amp_ps": float(tau_fit * 1e-6 / C0 * 1e12),
          "field_enhancement_fdtd": enh, "field_enhancement_pred": float(enh_pred * enh_now_factor),
          "field_enhancement_pred_steady_state": float(enh_pred), "field_enhancement_pred_from_calibration": float(enh_pred_cal),
          "field_enhancement_charge_factor_at_end": float(enh_now_factor), "charge_fraction_energy_at_end": float(E[-1] / E_ss),
          "buildup_energy_t": tl.tolist(), "buildup_energy": E.tolist()})
log(f"B5 CW at {1e3/f_res:.2f} nm: bent-mode radius peak {r_peak:.2f} µm, mean {r_mean:.2f} µm -> L_eff = {L_eff:.2f} µm, bend-corrected FSR {R['fsr_nm_pred_bend_corrected']:.2f} nm")
log(f"B5 build-up: τ_amp fit {tau_fit:.0f} µm/c = {R['buildup_tau_amp_ps']:.1f} ps (Harminv Q predicts {tau_pred:.0f}) -> Q_buildup = {Q_buildup:.0f}; "
    f"|E_ring|²/|E_bus|² = {enh:.2f} at t = {tlist[-1]:.0f} (formula (1−t²)/(1−ta)² = {enh_pred:.2f} steady state × {enh_now_factor:.2f} charged = {enh_pred*enh_now_factor:.2f}); "
    f"energy at the end = {E[-1]/E_ss*100:.0f} % of the fitted steady state")

# energy figure
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(tl * 1e-6 / C0 * 1e12, E / E_ss, color=SERIES[0], label="FDTD: mean envelope² of E_z in the ring / fitted steady state")
ax.plot(tl * 1e-6 / C0 * 1e12, model(tl, *popt) / E_ss, "--", color=SERIES[1], label=f"U(1 − e^{{−t/τ}})² fit, τ = {tau_fit*1e-6/C0*1e12:.1f} ps → Q = τπf = {Q_buildup:.0f}")
ax.axvline(tau_pred * 1e-6 / C0 * 1e12, color=SERIES[3], ls=":", label=f"τ = Qλ/π from Harminv Q = {Q_harm_c:.0f}: {tau_pred*1e-6/C0*1e12:.1f} ps")
ax.set(xlabel="time after the laser turns on (ps)", ylabel="stored energy / steady state", ylim=(0, 1.1),
       title="The ring charges up like an RC circuit:\namplitude time constant τ = 2Q/ω = Qλ/(πc), energy ∝ (1 − e^{−t/τ})²")
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "B5_buildup_energy.png"); plt.close(fig)

# movie
vmax = float(np.percentile(np.abs(ez_final), 99.5))
fig, (axf, axe) = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
im = axf.imshow(frames[0].T, origin="lower", extent=[xs[0], xs[-1], ys2[0], ys2[-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
th = np.linspace(0, 2 * np.pi, 300)
for rad in (RADIUS - W / 2, RADIUS + W / 2):
    axf.plot(rad * np.cos(th), YC + rad * np.sin(th), color=PALETTE["ink2"], lw=0.6)
for yy in (yb - W / 2, yb + W / 2):
    axf.plot([xs[0], xs[-1]], [yy, yy], color=PALETTE["ink2"], lw=0.6)
ttl = axf.set_title("")
axf.set(xlabel="x (µm)", ylabel="y (µm)")
axe.plot(tl * 1e-6 / C0 * 1e12, E / E_ss, color=PALETTE["line"], lw=1.2)
ln, = axe.plot([], [], color=SERIES[0]); dot, = axe.plot([], [], "o", color=SERIES[1], ms=7)
axe.set(xlabel="time (ps)", ylabel="energy in the ring / steady state", ylim=(0, 1.05), title="stored energy builds up over ~τ = 2Q/ω")
fig.tight_layout()
def draw(i):
    im.set_data(frames[i].T)
    tp = tl[i] * 1e-6 / C0 * 1e12
    ttl.set_text(f"E_z at t = {tp:6.2f} ps  ({tl[i]:.0f} µm/c): CW laser on resonance at {1e3/f_res:.2f} nm, {E[i]/E_ss*100:3.0f} % charged")
    ln.set_data(tl[:i + 1] * 1e-6 / C0 * 1e12, E[:i + 1] / E_ss); dot.set_data([tp], [E[i] / E_ss])
    return im, ttl, ln, dot
anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
anim.save(OUT / "B5_buildup.mp4", writer=FFMpegWriter(fps=30, bitrate=2500))
plt.close(fig)
# contact sheet
idxs = [len(t_fine) + 2, 40, 70, 110, 170, len(frames) - 1]      # all in the coarse phase: the light has reached the ring in every panel
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
for A, i in zip(axs.ravel(), idxs):
    A.imshow(frames[i].T, origin="lower", extent=[xs[0], xs[-1], ys2[0], ys2[-1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    A.set(title=f"t = {tl[i]*1e-6/C0*1e12:.2f} ps, {E[i]/E_ss*100:.0f} % charged", xlabel="x (µm)", ylabel="y (µm)")
    A.grid(False)
fig.suptitle(f"B5 contact sheet: CW laser on resonance ({1e3/f_res:.1f} nm) — the ring fills up (% = stored energy / fitted steady state) while the bus field stays weak")
fig.tight_layout(); fig.savefig(OUT / "B5_buildup_frames.png"); plt.close(fig)

R["runtime_s"] = time.time() - T0
with open(OUT / "B_results.json", "w") as f:
    json.dump(R, f, indent=2, default=float)
log(f"B done -> out/B_results.json")
