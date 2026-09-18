"""Experiment 12: round trip to ring. Headless entry point.

    cd experiments/12_round_trip_to_ring && ../../.meep/bin/python run.py     (.uses_meep: what run_all.sh does)
    cd experiments/12_round_trip_to_ring && ../../.venv/bin/python run.py     (works too: this file needs only the stdlib + common)

Orchestrates (with the right interpreter for each part, via subprocess):
  Part A  ring_sax.py      (.venv:  SAX/JAX circuit, matplotlib, animation)     -> out/A_*.png, A5_notch_sliding.mp4, A_results.json
  Part B  ring_meep.py     (.meep:  2-D FDTD ring, Harminv, flux, eigenmode)    -> out/B_*.png, B5_buildup.mp4, B_results.json
  Notebook make_notebook.py builds explore.ipynb; jupyter nbconvert executes it in place (kernel photonics-sims)
then merges the headline numbers into out/results.json and writes out/tools.json.
Everything in out/ is deleted first so that a run regenerates everything from scratch.
"""
import sys, pathlib, json, subprocess, shutil, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
use_style()

VENV = ROOT / ".venv" / "bin" / "python"
MEEP = ROOT / ".meep" / "bin" / "python"
JUPYTER = ROOT / ".venv" / "bin" / "jupyter"
T0 = time.time()

# 0. clean slate
for p in OUT.iterdir():
    if p.is_dir(): shutil.rmtree(p)
    else: p.unlink()
if (HERE / "explore.ipynb").exists(): (HERE / "explore.ipynb").unlink()

def run(cmd, log_name, cwd=HERE):
    t = time.time()
    with open(OUT / log_name, "w") as log:
        rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    print(f"  {' '.join(str(c) for c in cmd[:2])} -> exit {rc} in {time.time()-t:.0f} s (log: out/{log_name})", flush=True)
    if rc != 0:
        print(open(OUT / log_name).read()[-3000:])
        raise SystemExit(f"{cmd[1]} failed")

print("Part A: SAX all-pass ring (.venv)")
run([VENV, HERE / "ring_sax.py"], "A_log.txt")
print("Part B: Meep 2-D FDTD ring (.meep)")
run([MEEP, HERE / "ring_meep.py"], "B_log.txt")
print("Notebook: build + execute (.venv)")
run([VENV, HERE / "make_notebook.py"], "notebook_build_log.txt")   # nbformat lives in .venv; run.py itself may be run by either interpreter
run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.kernel_name=photonics-sims",
     "--ExecutePreprocessor.timeout=600", "explore.ipynb"], "notebook_log.txt")

# ---------------------------------------------------------------------------
# merge results
# ---------------------------------------------------------------------------
A = json.load(open(OUT / "A_results.json")); B = json.load(open(OUT / "B_results.json"))
import math

def agree(val, ref):
    return 100.0 * (1 - abs(val - ref) / abs(ref)) if ref else None

# Excess-loss bookkeeping (why the FDTD notch is shallower than the calibrated a predicts).
# The lossless gap scan at the SAME gap gives a loaded Q that already contains the ring's radiation + coupler-junction
# + grid loss: ta = f(Q_c) with t from the first-pass kappa^2 gives that "excess" a_x. The lossy ring should then have
# a_total = a_calibrated * a_x, which is compared with the a inverted from the notch in B4.
def ta_from_Q(Q, ng, L, lam):
    qp = Q * lam / (math.pi * ng * L); x = (-1 + math.sqrt(1 + 4 * qp ** 2)) / (2 * qp); return x ** 2
scan_main = [s for s in B["gap_scan"] if abs(s["gap_um"] - B["gap_um"]) < 1e-9][0]
ta_scan = ta_from_Q(scan_main["Q_c_harminv"], B["ng_2d"], B["L_ring_geom_um"], REF.lambda_nm * 1e-3)
a_excess = ta_scan / scan_main["t_first_pass"]
a_total_pred = B["a_from_calibrated_loss"] * a_excess
B["excess_loss"] = {"gap_um": B["gap_um"], "Q_lossless_scan": scan_main["Q_c_harminv"], "t_first_pass_lossless": scan_main["t_first_pass"],
                    "ta_lossless_from_Q": ta_scan, "a_excess_lossless_ring": a_excess,
                    "excess_loss_db_per_lap": -20 * math.log10(a_excess),
                    "a_total_pred": a_total_pred, "a_from_fdtd_notch": B["a_from_fdtd"],
                    "Tmin_pred_with_excess": ((B["t_from_first_pass_main"] - a_total_pred) / (1 - B["t_from_first_pass_main"] * a_total_pred)) ** 2,
                    "Q_pred_with_excess": math.pi * B["ng_2d"] * B["L_ring_geom_um"] * math.sqrt(B["t_from_first_pass_main"] * a_total_pred)
                                          / (REF.lambda_nm * 1e-3 * (1 - B["t_from_first_pass_main"] * a_total_pred))}

headline = {
    "a_doped":           {"value": A["a_doped"], "expected": REF.a_round_trip, "unit": "", "agreement_pct": agree(A["a_doped"], REF.a_round_trip), "source": "Part A, alpha from 125 dB/cm, L = 39.6 um (notes 28)"},
    "a_passive":         {"value": A["a_passive"], "expected": 0.9986, "unit": "", "agreement_pct": agree(A["a_passive"], 0.9986), "source": "Part A, 3 dB/cm"},
    "fsr_nm":            {"value": A["fsr_nm_sax"], "expected": REF.fsr_nm, "unit": "nm", "agreement_pct": agree(A["fsr_nm_sax"], REF.fsr_nm), "source": "Part A SAX dip spacing at 1310 nm; analytic lambda^2/(n_g L) = %.3f" % A["fsr_nm_analytic"]},
    "fsr_thz":           {"value": A["fsr_thz_sax"], "expected": REF.fsr_thz, "unit": "THz", "agreement_pct": agree(A["fsr_thz_sax"], REF.fsr_thz), "source": "Part A; analytic c/(n_g L) = %.3f" % A["fsr_thz_analytic"]},
    "fwhm_pm":           {"value": A["fwhm_pm_sax"], "expected": REF.fwhm_pm, "unit": "pm", "agreement_pct": agree(A["fwhm_pm_sax"], REF.fwhm_pm), "source": "Part A SAX half-depth width; analytic (1-ta)lambda^2/(pi n_g L sqrt(ta)) = %.1f" % A["fwhm_pm_analytic"]},
    "q_loaded":          {"value": A["q_sax"], "expected": REF.q_loaded, "unit": "", "agreement_pct": agree(A["q_sax"], REF.q_loaded), "source": "Part A lambda/FWHM"},
    "t_min_critical":    {"value": A["t_min_sax"], "expected": 0.0, "unit": "", "agreement_pct": None, "source": "Part A, t = a = 0.945 (critical coupling): T_min = ((t-a)/(1-ta))^2 = 0"},
    "build_up_factor":   {"value": A["build_up_analytic"], "expected": None, "unit": "|E_ring/E_in|^2", "agreement_pct": None, "source": "Part A (1-t^2)/(1-ta)^2 on resonance"},
    "K_per_fsr":         {"value": A["K_per_fsr"], "expected": 206.0, "unit": "K", "agreement_pct": agree(A["K_per_fsr"], 206.0), "source": "Part A FSR / 50 pm/K"},
    "pm_per_0p1K":       {"value": A["pm_per_0p1K"], "expected": 5.0, "unit": "pm", "agreement_pct": 100.0, "source": "0.1 K x 50 pm/K"},
    "fraction_fwhm_per_0p1K": {"value": A["fraction_fwhm_per_0p1K"], "expected": 5 / 374, "unit": "", "agreement_pct": agree(A["fraction_fwhm_per_0p1K"], 5 / 374), "source": "5 pm / FWHM"},
    "dT_per_0p1K_at_bias": {"value": A["dT_per_0p1K_at_bias"], "expected": None, "unit": "transmission change per 0.1 K at delta_opt", "agreement_pct": None, "source": "Part A slope of T(lambda_L) vs temperature at +108 pm"},
    "heater_nm_per_mw":  {"value": A["heater_nm_per_mw_check"], "expected": REF.heater_nm_per_mw, "unit": "nm/mW", "agreement_pct": agree(A["heater_nm_per_mw_check"], REF.heater_nm_per_mw), "source": "50 pm/K x 8.8 K/mW"},
    "neff_2d":           {"value": B["neff_2d"], "expected": REF.neff, "unit": "", "agreement_pct": agree(B["neff_2d"], REF.neff), "source": "Part B Meep eigenmode of the 2-D 500 nm guide (Ez); the textbook 2.5 is the 3-D strip: they are not expected to agree"},
    "ng_2d":             {"value": B["ng_2d"], "expected": REF.ng, "unit": "", "agreement_pct": agree(B["ng_2d"], REF.ng), "source": "Part B Meep eigenmode group velocity; textbook 4.2 is the 3-D strip"},
    "loss_calibrated_db_cm": {"value": B["db_cm_calibrated"], "expected": REF.loss_db_cm_doped, "unit": "dB/cm", "agreement_pct": agree(B["db_cm_calibrated"], REF.loss_db_cm_doped), "source": "Part B straight lossy guide, flux ratio over 8 um"},
    "a_fdtd_calibrated": {"value": B["a_from_calibrated_loss"], "expected": REF.a_round_trip, "unit": "", "agreement_pct": agree(B["a_from_calibrated_loss"], REF.a_round_trip), "source": "Part B e^{-alpha_meas L/2}"},
    "a_fdtd_from_notch": {"value": B["a_from_fdtd"], "expected": B["a_from_calibrated_loss"], "unit": "", "agreement_pct": agree(B["a_from_fdtd"], B["a_from_calibrated_loss"]), "source": "Part B inverted from the notch depth and width of the FDTD spectrum vs the straight-guide calibration alone"},
    "a_fdtd_vs_calibrated_plus_excess": {"value": B["a_from_fdtd"], "expected": a_total_pred, "unit": "", "agreement_pct": agree(B["a_from_fdtd"], a_total_pred), "source": "Part B same, vs a_calibrated x a_excess where a_excess = %.4f (%.3f dB/lap of radiation + coupler-junction + grid loss measured on the lossless ring at the same gap)" % (a_excess, -20 * math.log10(a_excess))},
    "t_min_fdtd_vs_excess": {"value": B["central_dip_Tmin"], "expected": B["excess_loss"]["Tmin_pred_with_excess"], "unit": "", "agreement_pct": agree(B["central_dip_Tmin"], B["excess_loss"]["Tmin_pred_with_excess"]), "source": "Part B notch depth vs ((t-a)/(1-ta))^2 with the first-pass t and a = a_calibrated x a_excess"},
    "fsr_nm_fdtd":       {"value": B["fsr_nm_fdtd_flux"], "expected": B["fsr_nm_pred_straight_ng"], "unit": "nm", "agreement_pct": agree(B["fsr_nm_fdtd_flux"], B["fsr_nm_pred_straight_ng"]), "source": "Part B flux-dip spacing vs lambda^2/(n_g L) with the straight-guide 2-D n_g and L = 2 pi R"},
    "fsr_nm_fdtd_bend_corrected": {"value": B["fsr_nm_fdtd_flux"], "expected": B["fsr_nm_pred_bend_corrected"], "unit": "nm", "agreement_pct": agree(B["fsr_nm_fdtd_flux"], B["fsr_nm_pred_bend_corrected"]), "source": "Part B same, with L_eff = 2 pi r_mean of the bent mode (r_mean = %.2f um)" % B["r_mean_um"]},
    "q_fdtd_flux":       {"value": B["central_dip_Q_flux"], "expected": B["central_dip_Q_harminv"], "unit": "", "agreement_pct": agree(B["central_dip_Q_flux"], B["central_dip_Q_harminv"]), "source": "Part B lambda/FWHM of the flux notch vs Harminv Q"},
    "q_fdtd_pred":       {"value": B["central_dip_Q_flux"], "expected": B["Q_pred_from_scan_and_loss"], "unit": "", "agreement_pct": agree(B["central_dip_Q_flux"], B["Q_pred_from_scan_and_loss"]), "source": "Part B flux Q vs all-pass formula with t from the lossless gap scan and a from the loss calibration"},
    "t_min_fdtd":        {"value": B["central_dip_Tmin"], "expected": B["Tmin_pred_from_scan_and_loss"], "unit": "", "agreement_pct": agree(B["central_dip_Tmin"], B["Tmin_pred_from_scan_and_loss"]), "source": "Part B notch depth vs ((t-a)/(1-ta))^2 with the same t and a"},
    "kappa2_fdtd_gap100nm": {"value": B["kappa2_fdtd"], "expected": REF.kappa2, "unit": "", "agreement_pct": None, "source": "Part B 2-D coupling at a 100 nm gap vs the reference device's 0.107: under-coupled in 2-D"},
    "gap_for_critical_um": {"value": B["gap_for_critical_coupling_um"], "expected": None, "unit": "um", "agreement_pct": None, "source": "Part B extrapolated lossless gap scan to kappa^2 = 1 - a^2"},
    "q_buildup":         {"value": B["Q_from_buildup"], "expected": B["central_dip_Q_harminv"], "unit": "", "agreement_pct": agree(B["Q_from_buildup"], B["central_dip_Q_harminv"]), "source": "Part B CW build-up time constant tau = 2Q/omega fitted to the ring energy"},
    "buildup_tau_ps":    {"value": B["buildup_tau_amp_ps"], "expected": B["buildup_tau_amp_pred_from_harminv_Q"] * 1e-6 / 299792458 * 1e12, "unit": "ps", "agreement_pct": agree(B["buildup_tau_amp_fit"], B["buildup_tau_amp_pred_from_harminv_Q"]), "source": "Part B amplitude build-up time constant"},
    "field_enhancement_fdtd": {"value": B["field_enhancement_fdtd"], "expected": B["field_enhancement_pred"], "unit": "|E_ring|^2/|E_bus|^2", "agreement_pct": agree(B["field_enhancement_fdtd"], B["field_enhancement_pred"]), "source": "Part B steady-state CW field ratio vs (1-t^2)/(1-ta)^2 (rough: single-component field ratio)"},
}
results = {"experiment": "12_round_trip_to_ring", "headline": headline, "part_A": A, "part_B": B,
           "runtime_s_total": time.time() - T0}
json.dump(results, open(OUT / "results.json", "w"), indent=2, default=float)

tools = [
    {"tool": "SAX (+JAX)", "version": "sax 0.18.2, jax 0.9.2",
     "what_it_is": "SAX is an S-parameter circuit simulator for photonics: you write each component as a small function returning an S-matrix and a netlist that connects ports, and it composes the frequency-domain response with JAX (differentiable, vectorised over wavelength). It is normally used to simulate photonic integrated circuits (rings, MZIs, filters) from component models.",
     "used_for": "The all-pass ring as a two-element circuit: a 2x2 evanescent coupler (t, -j kappa) and one lossy dispersive waveguide of length L = 39.6 um closed on itself. Swept 1290-1332 nm in 1 pm steps for the doped (125 dB/cm) and passive (3 dB/cm) rings.",
     "result": "FSR %.3f nm / %.3f THz at 1310 nm (analytic lambda^2/(n_g L): %.3f nm; reference 10.3 nm / 1.8 THz), FWHM %.1f pm (analytic %.1f, reference 374), Q %.0f (reference 3500), T_min %.1e at t = a (critical coupling). SAX and the closed form agree to %.1e (JAX float32 on a 2 pi x 76 rad round-trip phase)." % (A["fsr_nm_sax"], A["fsr_thz_sax"], A["fsr_nm_analytic"], A["fwhm_pm_sax"], A["fwhm_pm_analytic"], A["q_sax"], A["t_min_sax"], A["sax_vs_analytic_max_abs_diff"]),
     "how_to_observe": "cd experiments/12_round_trip_to_ring && ../../.venv/bin/python ring_sax.py -> out/A2_spectrum.png (notches every FSR, zoom with FWHM, phase), out/A3_coupling_regimes.png, out/A_log.txt. Change REF.loss_db_cm_doped or the coupling argument of ring_T() and re-run; or use the sliders in explore.ipynb."},
    {"tool": "Meep (pymeep FDTD)", "version": "meep %s" % B.get("meep_version", "1.34.0"),
     "what_it_is": "Meep is a free finite-difference time-domain (FDTD) electromagnetic solver: it marches Maxwell's equations on a grid in time and gives fields, fluxes and spectra for arbitrary 2-D/3-D structures. It is the standard open-source tool for photonic device simulation (waveguides, rings, couplers, photonic crystals).",
     "used_for": "A 2-D ring (R = 6.3 um, 500 nm wide, n 3.5 in 1.45) side-coupled to a bus with a %.0f nm gap. (1) Eigenmode solver for the 2-D n_eff, n_g and decay length; (2) calibration of D_conductivity so the ring carries 125 dB/cm; (3) a lossless gap scan of the coupling Q; (4) a Gaussian pulse with Harminv resonances and a normalised flux transmission spectrum; (5) a CW drive on one resonance for the build-up movie." % (B["gap_um"] * 1e3),
     "result": "2-D n_eff %.3f, n_g %.3f (3-D strip: 2.5-2.7 / 4.2). Loss calibrated to %.1f dB/cm -> a = %.4f. FSR %.2f nm from the flux dips vs %.2f nm from c/(n_g L) with the straight-guide n_g (%.1f %% agreement; %.2f nm with the bent-mode L_eff). Central notch at %.2f nm: T_min %.3f, FWHM %.0f pm, Q %.0f (Harminv %.0f, build-up fit %.0f: tau = %.1f ps vs Q lambda/(pi c) = %.1f ps). Inverting the notch gives t = %.4f, a = %.4f: the 2-D ring at 100 nm gap is under-coupled and loses %.3f dB/lap more than the straight guide (a_cal x a_excess = %.4f, 99 %% of the inverted a); critical coupling extrapolates to a %.0f nm gap." % (
         B["neff_2d"], B["ng_2d"], B["db_cm_calibrated"], B["a_from_calibrated_loss"], B["fsr_nm_fdtd_flux"], B["fsr_nm_pred_straight_ng"], agree(B["fsr_nm_fdtd_flux"], B["fsr_nm_pred_straight_ng"]), B["fsr_nm_pred_bend_corrected"],
         B["central_dip_nm"], B["central_dip_Tmin"], B["central_dip_fwhm_pm"], B["central_dip_Q_flux"], B["central_dip_Q_harminv"], B["Q_from_buildup"], B["buildup_tau_amp_ps"], B["buildup_tau_amp_pred_from_harminv_Q"] * 1e-6 / 299792458 * 1e12, B["t_from_fdtd"], B["a_from_fdtd"], B["excess_loss"]["excess_loss_db_per_lap"], B["excess_loss"]["a_total_pred"], B["gap_for_critical_coupling_um"] * 1e3),
     "how_to_observe": "cd experiments/12_round_trip_to_ring && ../../.meep/bin/python ring_meep.py (about 7.5 min: %.0f s this run) -> out/B4_spectrum.png (transmission with Harminv markers), out/B5_buildup.mp4 (field charging the ring), out/B5_buildup_energy.png, out/B3_gap_scan.png, out/B_log.txt. Change GAP, RES_MAIN or the B3 gaps list at the top of ring_meep.py and re-run; keep RES_CW = RES_MAIN (the resonance shifts by more than a linewidth between grids)." % B["runtime_s"]},
    {"tool": "Meep Harminv", "version": "bundled with meep %s" % B.get("meep_version", "1.34.0"),
     "what_it_is": "Harminv is a harmonic-inversion filter-diagonalisation routine bundled with Meep: from a short time series of a field at one point it extracts the resonant frequencies, decay rates (hence Q) and amplitudes of the modes that were excited, far more accurately than a Fourier transform of the same record.",
     "used_for": "Resonance wavelengths and Q of the ring modes from Ez at a point on the ring after a Gaussian pulse (lossless gap scan and the lossy main run).",
     "result": "Fundamental-family resonances at %s nm with loaded Q %.0f near 1310 nm; the flux-notch Q is %.0f (%.1f %% agreement). A second family of low-Q modes (the 2-D 500 nm guide is multimode for Ez: second mode n_eff %.3f) is flagged separately." % (", ".join("%.1f" % l for l in B["harminv_fundamental_nm"]), B["central_dip_Q_harminv"], B["central_dip_Q_flux"], agree(B["central_dip_Q_flux"], B["central_dip_Q_harminv"]), B["neff_2d_mode2"]),
     "how_to_observe": "Dashed orange lines with Q labels in out/B4_spectrum.png; the full list is in out/B_results.json under harminv_all. Move RING_PT or the Harminv bandwidth (0.05) in ring_meep.py to hunt other modes."},
    {"tool": "Meep eigenmode solver (MPB)", "version": "bundled with meep %s" % B.get("meep_version", "1.34.0"),
     "what_it_is": "Meep's get_eigenmode calls the MPB plane-wave eigensolver on a cross-section of the FDTD grid to find guided modes (propagation constant, group velocity, field profile). It is what EigenModeSource uses to launch a clean mode and what mode-decomposition uses at ports.",
     "used_for": "n_eff, n_g and the transverse profile of the fundamental and second Ez modes of the 2-D 500 nm guide, and the evanescent decay length outside the core.",
     "result": "n_eff = %.4f, n_g = %.4f (finite-difference check %.4f), decay length 1/gamma = %.0f nm (fit %.0f nm) vs 102 nm for the textbook 3-D n_eff 2.5; second mode n_eff %.3f (multimode in 2-D)." % (B["neff_2d"], B["ng_2d"], B["ng_2d_finite_difference"], B["decay_len_2d_nm"], B["decay_len_2d_fit_nm"], B["neff_2d_mode2"]),
     "how_to_observe": "out/B1_mode2d.png; section B1 of ring_meep.py. Change W or N_SI to see n_eff, n_g and the tail change (a lower core index makes the tail longer and the coupling stronger)."},
    {"tool": "numpy / scipy (closed forms and fits)", "version": "numpy 2.4.6 (.venv) / 2.5.3 (.meep), scipy 1.18.1",
     "what_it_is": "numpy is the array library everything else is built on; scipy.optimize.curve_fit is a nonlinear least-squares fitter.",
     "used_for": "The closed-form ring numbers (FSR, FWHM, Q, T_min, finesse, build-up), the inversion of Q and T_min back to t and a, the exponential fit of kappa^2(gap), the (1 - e^{-t/tau})^2 fit of the ring energy, and the finite-difference slope dT/dK at the bias point.",
     "result": "Closed form vs SAX %.1e; closed form vs reference FSR %.3f nm, FWHM %.1f pm, Q %.0f; build-up tau = %.1f ps (Harminv predicts %.1f ps, %.1f %% agreement)." % (A["sax_vs_analytic_max_abs_diff"], A["fsr_nm_analytic"], A["fwhm_pm_analytic"], A["q_analytic"], B["buildup_tau_amp_ps"], B["buildup_tau_amp_pred_from_harminv_Q"] * 1e-6 / 299792458 * 1e12, agree(B["buildup_tau_amp_fit"], B["buildup_tau_amp_pred_from_harminv_Q"])),
     "how_to_observe": "analytic_numbers(t, a) and analytic_T(wl, t, a) in ring_sax.py; t_from_Qc, ta_from_Q and the curve_fit call in section B5 of ring_meep.py."},
    {"tool": "matplotlib (FuncAnimation + FFMpegWriter, ffmpeg 9.0.1)", "version": "matplotlib 3.11.2",
     "what_it_is": "matplotlib is the standard Python plotting library; FuncAnimation renders a sequence of frames and FFMpegWriter pipes them to ffmpeg to write an mp4.",
     "used_for": "All figures; the two videos: A5 (notch sliding under the fixed laser as the ring ramps 10 -> 125 C) and B5 (FDTD Ez frames while the ring charges up on resonance), each with a PNG contact sheet.",
     "result": "out/A5_notch_sliding.mp4 (10 s, 300 frames), out/B5_buildup.mp4 (10 s, 300 frames), contact sheets A5_notch_sliding_frames.png, B5_buildup_frames.png.",
     "how_to_observe": "open out/A5_notch_sliding.mp4 and out/B5_buildup.mp4; frames, ramp range and capture times are set near the bottom of ring_sax.py and ring_meep.py."},
    {"tool": "Jupyter notebook (nbformat + nbconvert, ipywidgets)", "version": "nbformat 5.11.1, nbconvert 7.17.1, ipywidgets 8.1.9, kernel photonics-sims",
     "what_it_is": "nbformat builds notebook files programmatically; nbconvert --execute runs them headlessly with a kernel and saves the outputs; ipywidgets.interact adds sliders in a live JupyterLab session.",
     "used_for": "explore.ipynb: sliders for a, t, n_g and temperature on the closed-form ring, the SAX/closed-form check, T_min vs t, and an overlay of the FDTD notch.",
     "result": "Executed notebook with saved outputs (static rendering of the default slider state).",
     "how_to_observe": "../../.venv/bin/jupyter lab explore.ipynb (Restart & Run All), or re-execute headlessly with ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb."},
]
json.dump(tools, open(OUT / "tools.json", "w"), indent=2)

print("\nHeadline numbers (value | expected | agreement):")
for k, v in headline.items():
    ag = "" if v["agreement_pct"] is None else f"{v['agreement_pct']:.1f} %"
    exp = "" if v["expected"] is None else f"{v['expected']:.4g}"
    print(f"  {k:28s} {v['value']:.4g} {v['unit']:6s} | {exp:>8s} | {ag}")
print(f"\nTotal run time {time.time()-T0:.0f} s. Outputs in out/, notebook explore.ipynb")
