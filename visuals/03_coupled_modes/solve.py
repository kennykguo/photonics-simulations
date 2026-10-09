"""Stage 1: every mode solve and every number the step figures use.

Writes out/cache/solved.npz (fields on grids) and out/results.json (numbers).
About 4 min. run.py calls this only if the cache is missing (or with --resolve).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import cmt
from cmt import GAP, H, K0, LAM, W

HERE = Path(__file__).resolve().parent
CACHE = HERE / "out" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

XS = np.round(np.arange(-1.30, 1.3001, 0.01), 4)
YS = np.round(np.arange(-0.55, 0.5501, 0.01), 4)
XL = np.round(np.arange(-1.30, 1.3001, 0.002), 4)      # fine line cut at y = 0
GAPS = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40]


def main():
    t0 = time.time()
    R = {"lam_um": LAM, "w_um": W, "h_um": H, "gap_um": GAP, "n_si": cmt.N_SI, "n_ox": cmt.N_OX,
         "k0": K0}
    D = {"xs": XS, "ys": YS, "xl": XL}
    p = cmt.build(GAP, res=0.015)
    R["mesh_elements"] = int(p.mesh.nelements)
    xa, xb = p.xa, p.xb
    D["xa"], D["xb"] = xa, xb

    # ---------------- scalar model: the notes' equation ----------------
    nA, uA = cmt.scalar_modes(p, True, False, 1)
    nB, uB = cmt.scalar_modes(p, False, True, 2)            # B0 and B1 (antisymmetric)
    ns, us = cmt.scalar_modes(p, True, True, 2)
    uA, uB0, uB1 = uA[:, 0], uB[:, 0], uB[:, 1]
    up, um = us[:, 0], us[:, 1]
    # sign of the odd supermode: positive in A, so A = (u+ + u-)/sqrt2 by construction
    if cmt.line_scalar(p, um, np.array([xa]))[0] < 0:
        um = -um
    K, M, N = cmt._scalar_mats(p)
    s = {}
    s["n_A"], s["n_B0"], s["n_B1"] = float(nA[0]), float(nB[0]), float(nB[1])
    s["n_plus"], s["n_minus"] = float(ns[0]), float(ns[1])
    s["kappa_exact"] = math.pi * (ns[0] - ns[1]) / LAM
    s["kappa_overlap"] = float(cmt.scalar_kappa_overlap(p, uA, uB0, nA[0]))
    s["L_c_exact"] = math.pi / (2 * s["kappa_exact"])
    # projection of A's leftover onto B's two scalar modes, and the A.B non-orthogonality
    DB = N(cmt.eps_map(p.s0, delta_only="B"))
    s["proj_B0"] = float(uB0 @ DB @ uA)
    s["proj_B1"] = float(uB1 @ DB @ uA)
    s["overlap_uA_uB"] = float(uA @ M @ uB0)
    s["c_plus"], s["c_minus"] = float(up @ M @ uA), float(um @ M @ uA)
    # fraction of A's |u|^2 that lies inside core B
    DBm = N(cmt.eps_map(p.s0, delta_only="B") / cmt.DN2)
    s["frac_A_in_B"] = float(uA @ DBm @ uA)
    R["scalar"] = s

    D["uA"] = cmt.sample_scalar(p, uA, XS, YS)
    D["uB0"] = cmt.sample_scalar(p, uB0, XS, YS)
    D["uB1"] = cmt.sample_scalar(p, uB1, XS, YS)
    D["uP"] = cmt.sample_scalar(p, up, XS, YS)
    D["uM"] = cmt.sample_scalar(p, um, XS, YS)
    D["uA_line"] = cmt.line_scalar(p, uA, XL)
    D["uB0_line"] = cmt.line_scalar(p, uB0, XL)
    D["uB1_line"] = cmt.line_scalar(p, uB1, XL)
    # isolated-A scalar mode with only A present (same mesh), for step 1, is uA itself
    res = cmt.scalar_residual(p, uA, nA[0])
    D["residual"] = cmt.sample_scalar(p, res, XS, YS)
    inB = (np.abs(XS[None, :] - xb) <= W / 2 - 0.02) & (np.abs(YS[:, None]) <= H / 2 - 0.02)
    farB = ~((np.abs(XS[None, :] - xb) <= W / 2 + 0.05) & (np.abs(YS[:, None]) <= H / 2 + 0.05))
    pred = cmt.K0**2 * cmt.DN2 * D["uA"]
    s["residual_over_prediction_in_B_median"] = float(np.median(D["residual"][inB] / pred[inB]))
    s["residual_outside_B_max_rel"] = float(np.abs(D["residual"][farB]).max() / np.abs(D["residual"][inB]).max())
    # decay of the scalar tail between the cores
    sel = (XL > xa + W / 2 + 0.04) & (XL < xb - W / 2 - 0.01)
    s["gamma_tail_fit"] = float(-np.polyfit(XL[sel], np.log(np.abs(D["uA_line"][sel])), 1)[0])

    # ---------------- vector model: full Maxwell ----------------
    vA = cmt.classify(cmt.vector_modes(p, True, False, 3))[0][0]
    teB, tmB = cmt.classify(cmt.vector_modes(p, False, True, 6))
    tePair, _ = cmt.classify(cmt.vector_modes(p, True, True, 4))
    vB0, vB1, vTM = teB[0], teB[1], tmB[0]
    vP, vM = tePair[0], tePair[1]
    v = {"n_A": vA.n_eff, "n_B_TE0": vB0.n_eff, "n_B_TE1": vB1.n_eff, "n_B_TM0": vTM.n_eff,
         "n_plus": vP.n_eff, "n_minus": vM.n_eff,
         "te_fraction_A": vA.te_fraction, "te_fraction_plus": vP.te_fraction,
         "te_fraction_minus": vM.te_fraction}
    v["kappa_exact"] = math.pi * (vP.n_eff - vM.n_eff) / LAM
    v["kappa_overlap"], _ = cmt.vector_kappa_overlap(p, vA, vB0)
    v["L_c_exact"] = math.pi / (2 * v["kappa_exact"])
    v["L_c_overlap"] = math.pi / (2 * v["kappa_overlap"])
    for lab, mb in (("TE0", vB0), ("TE1", vB1), ("TM0", vTM)):
        v[f"proj_{lab}"] = cmt.vector_overlap_abs(p, vA, mb)
    v["overlap_A_B_power"] = abs(cmt.power_overlap(vA, vB0))
    R["vector"] = v

    ref = (xa, 0.0)
    for key, m, r in (("A", vA, ref), ("B0", vB0, (xb, 0.0)), ("B1", vB1, (xb + 0.12, 0.0)),
                      ("P", vP, ref), ("M", vM, ref)):
        for comp, arr in cmt.sample_vector(m, XS, YS, phase_ref=r).items():
            D[f"v{key}_{comp}"] = arr
    # Expansion coefficients A = c+ e+ + c- e-, from the very fields that are drawn
    # (each sampled mode carries its own phase convention, so the coefficients must
    # be computed from the samples, not from the solver's internal phases).
    dA_ = (XS[1] - XS[0]) * (YS[1] - YS[0])

    def ov(a, b):     # power overlap <a|b> = 1/2 int (Ea* x Hb + Eb x Ha*) . z
        return 0.5 * dA_ * np.sum(np.conj(D[f"v{a}_Ex"]) * D[f"v{b}_Hy"] - np.conj(D[f"v{a}_Ey"]) * D[f"v{b}_Hx"]
                                  + D[f"v{b}_Ex"] * np.conj(D[f"v{a}_Hy"]) - D[f"v{b}_Ey"] * np.conj(D[f"v{a}_Hx"]))

    cp, cm = ov("P", "A") / ov("P", "P"), ov("M", "A") / ov("M", "M")
    v["power_in_plus"] = float(abs(ov("P", "A")) ** 2 / (ov("P", "P").real * ov("A", "A").real))
    v["power_in_minus"] = float(abs(ov("M", "A")) ** 2 / (ov("M", "M").real * ov("A", "A").real))
    recon = cp * D["vP_Ex"] + cm * D["vM_Ex"]
    v["launch_reconstruction_error"] = float(np.linalg.norm(recon - D["vA_Ex"]) / np.linalg.norm(D["vA_Ex"]))
    D["vA_Ex_line"] = np.real(cmt.line_vector(vA, XL, phase_ref=ref))
    D["vB_Ex_line"] = np.real(cmt.line_vector(vB0, XL, phase_ref=(xb, 0.0)))
    sel = (XL > xa + W / 2 + 0.04) & (XL < xb - W / 2 - 0.01)
    v["gamma_tail_fit"] = float(-np.polyfit(XL[sel], np.log(np.abs(D["vA_Ex_line"][sel])), 1)[0])

    # ---------------- propagation: coupled-mode ODE vs exact supermodes ----------------
    k = v["kappa_overlap"]
    zz = np.linspace(0, 2.2 * v["L_c_exact"], 1201)

    def rhs(z, y):
        a, b = y[0] + 1j * y[1], y[2] + 1j * y[3]
        da, db = -1j * k * b, -1j * k * a
        return [da.real, da.imag, db.real, db.imag]

    sol = solve_ivp(rhs, (zz[0], zz[-1]), [1, 0, 0, 0], t_eval=zz, rtol=1e-10, atol=1e-12)
    D["z"] = zz
    D["ode_A"] = sol.y[0] + 1j * sol.y[1]
    D["ode_B"] = sol.y[2] + 1j * sol.y[3]
    # exact: launch A = c+ e+ + c- e-, each supermode keeps its own beta.
    bp, bm = vP.beta, vM.beta
    D["sup_cp"], D["sup_cm"] = cp, cm      # from the sampled fields, above
    D["beta_p"], D["beta_m"] = bp, bm
    # half-plane power of the superposition (what the FDTD measures)
    dx, dy = XS[1] - XS[0], YS[1] - YS[0]
    left = XS < 0
    PA, PB = [], []
    for z in zz[::6]:
        ep, em = cp * np.exp(-1j * bp * z), cm * np.exp(-1j * bm * z)
        Ex = ep * D["vP_Ex"] + em * D["vM_Ex"]
        Ey = ep * D["vP_Ey"] + em * D["vM_Ey"]
        Hx = ep * D["vP_Hx"] + em * D["vM_Hx"]
        Hy = ep * D["vP_Hy"] + em * D["vM_Hy"]
        Sz = np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        PA.append(Sz[:, left].sum() * dx * dy)
        PB.append(Sz[:, ~left].sum() * dx * dy)
    PA, PB = np.array(PA), np.array(PB)
    D["z_half"] = zz[::6]
    D["half_PA"], D["half_PB"] = PA / (PA + PB), PB / (PA + PB)
    v["half_power_total_spread"] = float((PA + PB).std() / (PA + PB).mean())

    # ---------------- gap sweep ----------------
    sweep_file = CACHE / "sweep.json"
    sweep = json.loads(sweep_file.read_text()) if sweep_file.exists() else []
    for g in ([] if sweep else GAPS):
        t = time.time()
        sweep.append(cmt.kappa_at_gap(g, res=0.02))
        print(f"  gap {g*1e3:.0f} nm: vector exact {sweep[-1]['vector']['kappa_exact']:.4f}, "
              f"overlap {sweep[-1]['vector']['kappa_overlap']:.4f}, scalar exact "
              f"{sweep[-1]['scalar']['kappa_exact']:.4f}  ({time.time()-t:.0f} s)", flush=True)
    sweep_file.write_text(json.dumps(sweep, default=float))
    R["gap_sweep"] = sweep
    gs = np.array(GAPS)
    for model in ("vector", "scalar"):
        ke = np.array([s_[model]["kappa_exact"] for s_ in sweep])
        R[f"{model}_kappa_slope_per_um"] = float(-np.polyfit(gs, np.log(ke), 1)[0])

    R["runtime_s"] = time.time() - t0
    np.savez_compressed(CACHE / "solved.npz", **D)
    (HERE / "out" / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(f"solve done in {R['runtime_s']:.0f} s")
    return R


if __name__ == "__main__":
    main()
