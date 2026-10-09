"""TE versus TM in a 500 x 220 nm silicon strip at 1310 nm, from the full vector fields.

Run:   cd ~/photonics-simulations/visuals/02_te_vs_tm && ../../.venv/bin/python run.py
Out:   out/te_vs_tm_poster.png  (the one-page summary)
       out/1_cross_sections.png ... out/6_capstone.png  (each panel in detail)
       out/te_tm_travelling.mp4 / .gif  (the hybrid fields moving along the guide)
       out/results.json
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Rectangle

import tetm

warnings.filterwarnings("ignore")

LAM, W, H = 1.31, 0.50, 0.22
RING_L = 39.6
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

C_TE = "#2a78d6"     # blue   = quasi-TE everywhere
C_TM = "#eb6834"     # orange = quasi-TM everywhere
C_INK, C_MUTED = "#1a1a19", "#6b6a66"
C_SI, C_OX = "#c9d6e3", "#f3f5f7"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d8d6d0", "axes.grid": True, "grid.color": "#ecebe6",
    "grid.linewidth": 0.7, "text.color": C_INK, "axes.labelcolor": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "font.size": 10,
    "legend.frameon": False, "savefig.dpi": 140, "savefig.bbox": "tight",
})

t0 = time.time()
R = {}


def say(s):
    print(f"[{time.time() - t0:6.1f}s] {s}", flush=True)


# ===================================================================== solve
say("solving the strip (femwell, full vector, 4 modes)")
n_si, n_ox = float(tetm.ix.n_silicon(LAM)), float(tetm.ix.n_silica(LAM))
TE, TM = tetm.pick_te_tm(tetm.solve(LAM, W, H, num_modes=4, res=0.010))
DX = 0.0075
xs = np.arange(-0.81, 0.8101, DX)
ys = np.arange(-0.54, 0.5401, DX)
G = {"TE": tetm.sample(TE, xs, ys), "TM": tetm.sample(TM, xs, ys)}
M = {"TE": TE, "TM": TM}
X, Y = G["TE"]["X"], G["TE"]["Y"]
core = (np.abs(X) <= W / 2) & (np.abs(Y) <= H / 2)
epsmap = np.where(core, n_si**2, n_ox**2)
dA = DX**2

say(f"  quasi-TE0 n_eff = {TE.n_eff:.4f}  TE fraction {TE.te_fraction:.3f}")
say(f"  quasi-TM0 n_eff = {TM.n_eff:.4f}  TE fraction {TM.te_fraction:.3f}")

# slab references
slab = {p: tetm.slab_neff(LAM, H, n_si, n_ox, p) for p in ("TE", "TM")}


def stats(key):
    g, m = G[key], M[key]
    eparts = {c: float(np.sum(epsmap * np.abs(g[c]) ** 2) * dA) for c in ("Ex", "Ey", "Ez")}
    hparts = {c: float(np.sum(np.abs(g[c]) ** 2) * dA) for c in ("Hx", "Hy", "Hz")}
    es, hs = sum(eparts.values()), sum(hparts.values())
    e2 = sum(np.abs(g[c]) ** 2 for c in ("Ex", "Ey", "Ez"))
    peak = e2.max()
    # field intensity just outside the middle of a sidewall and of the top face
    iy0, ix0 = np.argmin(np.abs(ys)), np.argmin(np.abs(xs))
    ixw = np.argmin(np.abs(xs - (W / 2 + 0.008)))
    iyt = np.argmin(np.abs(ys - (H / 2 + 0.008)))
    # decay lengths of |E| outside the core, fitted 50-250 nm beyond each face
    def decay(line, coord, face):
        sel = (coord > face + 0.05) & (coord < face + 0.25)
        k = np.polyfit(coord[sel], np.log(np.abs(line[sel]) + 1e-30), 1)[0]
        return -1000 / k
    side = np.sqrt(e2[iy0, :])
    top = np.sqrt(e2[:, ix0])
    return {
        "n_eff": m.n_eff, "te_fraction": m.te_fraction,
        "energy_share_E": {k: v / es for k, v in eparts.items()},
        "energy_share_H": {k: v / hs for k, v in hparts.items()},
        "dneff_dnsi": m.dneff_dnsi,
        "E2_at_sidewall_over_peak": float(e2[iy0, ixw] / peak),
        "E2_at_topface_over_peak": float(e2[iyt, ix0] / peak),
        "decay_nm_beyond_sidewall": decay(side, xs, W / 2),
        "decay_nm_above_top": decay(top, ys, H / 2),
        "power_fraction_in_core": float(np.sum(np.abs(g["Ex"] * np.conj(g["Hy"])
                                                      - g["Ey"] * np.conj(g["Hx"]))[core])
                                        / np.sum(np.abs(g["Ex"] * np.conj(g["Hy"])
                                                        - g["Ey"] * np.conj(g["Hx"])))),
    }


R["TE"], R["TM"] = stats("TE"), stats("TM")
R["slab_220nm"] = slab
R["materials"] = {"lambda_um": LAM, "n_si": n_si, "n_sio2": n_ox,
                  "eps_ratio": (n_si / n_ox) ** 2}

# boundary jumps, measured
say("measuring the normal-field jumps at the faces")
jumps = {
    "TE_sidewall_Ex": tetm.face_jump(TE, 0, W / 2, n_si, n_ox),
    "TE_top_Ex_tangential": tetm.face_jump(TE, 1, H / 2, n_si, n_ox),
    "TM_top_Ey": tetm.face_jump(TM, 1, H / 2, n_si, n_ox),
    "TM_sidewall_Ey_tangential": tetm.face_jump(TM, 0, W / 2, n_si, n_ox),
}


def _probe_comp(mode, pts, comp):
    return np.real(tetm.probe_line(mode, pts)[comp])


# tangential components: measure them, they should NOT jump
def tang(mode, axis, face, comp):
    s = np.array([0.002, 0.004, 0.006, 0.008])
    if axis == 0:
        pin, pout = np.vstack([face - s, 0 * s]), np.vstack([face + s, 0 * s])
    else:
        pin, pout = np.vstack([0 * s, face - s]), np.vstack([0 * s, face + s])
    a = np.polyfit(s, _probe_comp(mode, pin, comp), 1)[1]
    b = np.polyfit(s, _probe_comp(mode, pout, comp), 1)[1]
    return b / a


R["jumps"] = {
    "expected_normal_ratio": (n_si / n_ox) ** 2,
    "TE_Ex_across_sidewall (normal)": jumps["TE_sidewall_Ex"][2],
    "TE_Ex_across_top (tangential)": tang(TE, 1, H / 2, 0),
    "TM_Ey_across_top (normal)": jumps["TM_top_Ey"][2],
    "TM_Ey_across_sidewall (tangential)": tang(TM, 0, W / 2, 1),
}
for k, v in R["jumps"].items():
    say(f"  {k:40s} {v:.3f}")


# ================================================================ helpers
def outline(ax, w=W, h=H, lw=1.3, color=C_INK):
    ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fill=False, ec=color, lw=lw, zorder=6))


def cross_section(ax, key, cbar=True):
    g = G[key]
    e = np.sqrt(sum(np.abs(g[c]) ** 2 for c in ("Ex", "Ey", "Ez")))
    im = ax.imshow(e, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                   cmap="magma_r", vmin=0, vmax=e.max(), interpolation="bilinear")
    sk = 6
    Xq, Yq = X[::sk, ::sk], Y[::sk, ::sk]
    U, V = np.real(g["Ex"][::sk, ::sk]), np.real(g["Ey"][::sk, ::sk])
    mag = np.hypot(U, V)
    keep = mag > 0.06 * mag.max()
    col = C_TE if key == "TE" else C_TM
    ax.quiver(Xq[keep], Yq[keep], U[keep], V[keep], color="white", scale=14,
              width=0.006, headwidth=3.6, zorder=4)
    ax.quiver(Xq[keep], Yq[keep], U[keep], V[keep], color=col, scale=14,
              width=0.0035, headwidth=3.6, zorder=5)
    outline(ax)
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.42, 0.42)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x, across the width, in the chip plane (um)")
    ax.set_ylabel("y, across the thickness (um)")
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("|E| (normalised)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return im


def cut_axes(key, axis):
    """Field and D = n^2 E along a horizontal (axis 0) or vertical (axis 1) cut."""
    g = G[key]
    comp = "Ex" if key == "TE" else "Ey"
    if axis == 0:
        coord = xs
        line = np.real(g[comp][np.argmin(np.abs(ys)), :])
        inside = np.abs(coord) <= W / 2
        half = W / 2
    else:
        coord = ys
        line = np.real(g[comp][:, np.argmin(np.abs(xs))])
        inside = np.abs(coord) <= H / 2
        half = H / 2
    eps = np.where(inside, n_si**2, n_ox**2)
    return coord, line, eps * line / (n_si**2), half, comp


def draw_cut(ax, key, axis, title):
    coord, e, d, half, comp = cut_axes(key, axis)
    col = C_TE if key == "TE" else C_TM
    normal = (key == "TE" and axis == 0) or (key == "TM" and axis == 1)
    ax.axvspan(-half, half, color=C_SI, alpha=0.55, zorder=0, lw=0)
    ax.plot(coord, e, color=col, lw=2.2, label=f"{comp} (the dominant E)")
    ax.plot(coord, d, color=C_INK, lw=1.3, ls="--",
            label=f"D = n" + "²" + f"{comp}, scaled")
    ax.set_xlim(coord[0] * 0.75, coord[-1] * 0.75)
    ax.set_ylim(-0.05, 1.18 if not normal else max(1.18, e.max() * 1.12))
    ax.set_xlabel(("x, across the width" if axis == 0 else "y, across the thickness") + " (um)")
    ax.set_title(title, fontsize=9.5, loc="left")
    if normal:
        k = "TE_Ex_across_sidewall (normal)" if key == "TE" else "TM_Ey_across_top (normal)"
        ax.annotate(f"jumps x{R['jumps'][k]:.2f}\n(theory (n$_{{Si}}$/n$_{{ox}}$)" + "²"
                    + f" = {R['jumps']['expected_normal_ratio']:.2f})",
                    xy=(half, e[np.argmin(np.abs(coord - half - 0.006))]),
                    xytext=(half + 0.07, e.max() * 0.93), fontsize=8.5, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1))
    else:
        k = "TE_Ex_across_top (tangential)" if key == "TE" else "TM_Ey_across_sidewall (tangential)"
        ax.text(half + 0.03, 0.88, f"continuous\n(ratio {R['jumps'][k]:.3f})",
                fontsize=8.5, color=col)
    ax.legend(loc="upper left", fontsize=7.5)


# ======================================================== figure 1: cross sections
say("figure 1: cross sections with E arrows")
fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"wspace": 0.28})
for ax, key in zip(axs, ("TE", "TM")):
    cross_section(ax, key)
    r = R[key]
    name = "quasi-TE0: E lies in the chip plane" if key == "TE" else "quasi-TM0: E points out of the chip"
    ax.set_title(f"{name}\nn$_{{eff}}$ = {r['n_eff']:.3f}   |   "
                 f"{r['power_fraction_in_core']*100:.0f} % of the power in the silicon   |   "
                 f"TE fraction {r['te_fraction']:.3f}",
                 fontsize=10, color=C_TE if key == "TE" else C_TM, loc="left")
fig.suptitle("Same 500 x 220 nm silicon strip, same 1310 nm light, two different modes. "
             "Arrows: the transverse electric field. Colour: |E|.", fontsize=12, y=1.0)
fig.savefig(OUT / "1_cross_sections.png")
plt.close(fig)

# ======================================================== figure 2: six components
say("figure 2: all six field components (why it is 'quasi')")
fig, axs = plt.subplots(2, 6, figsize=(17, 5.6))
for row, key in enumerate(("TE", "TM")):
    g, r = G[key], R[key]
    for col, c in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        ax = axs[row, col]
        f = g[c]
        part = np.imag(f) if c.endswith("z") else np.real(f)
        lim = max(np.abs(part).max(), 1e-12)
        ax.imshow(part, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                  cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="bilinear")
        outline(ax, lw=0.9)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.4, 0.4)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        share = (r["energy_share_E"] if c[0] == "E" else r["energy_share_H"])[c]
        peak = np.abs(f).max()
        z = " (imag: 90° behind)" if c.endswith("z") else ""
        ax.set_title(f"{c}{z}\npeak {peak:.2f}, {share*100:.1f} % of {'electric' if c[0]=='E' else 'magnetic'} energy",
                     fontsize=8.3, color=C_INK if share > 0.2 else C_MUTED)
        if col == 0:
            ax.set_ylabel("quasi-TE0" if key == "TE" else "quasi-TM0", fontsize=12,
                          color=C_TE if key == "TE" else C_TM, weight="bold")
fig.suptitle("All six components are non-zero in both modes: a rectangular wire has no pure TE "
             "or TM mode, hence 'quasi'.\nEach mode has one dominant E and one dominant H; "
             "the longitudinal Ez and Hz live at the faces the dominant field crosses or skims.",
             fontsize=11.5, y=1.04)
fig.savefig(OUT / "2_six_components.png")
plt.close(fig)

# ======================================================== figure 3: boundary jumps
say("figure 3: line cuts, the normal component jumps and D does not")
fig, axs = plt.subplots(2, 2, figsize=(13, 7.6), sharey="row")
draw_cut(axs[0, 0], "TE", 0, "quasi-TE, horizontal cut through the centre:\nEx is NORMAL to the sidewalls")
draw_cut(axs[0, 1], "TE", 1, "quasi-TE, vertical cut through the centre:\nEx is TANGENTIAL to the top and bottom")
draw_cut(axs[1, 0], "TM", 0, "quasi-TM, horizontal cut through the centre:\nEy is TANGENTIAL to the sidewalls")
draw_cut(axs[1, 1], "TM", 1, "quasi-TM, vertical cut through the centre:\nEy is NORMAL to the top and bottom")
for ax in axs[:, 0]:
    ax.set_ylabel("field (normalised to its peak)")
fig.suptitle("The rule that decides everything: tangential E is continuous, normal E jumps by "
             "n" + "²" + "$_{in}$/n" + "²" + "$_{out}$ (because D = n" + "²" + "E is continuous).\n"
             "The TM field crosses the two LARGE faces, so it is thrown into the silica above and below. "
             "The TE field only crosses the two short sidewalls.",
             fontsize=11.5, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "3_boundary_jumps.png")
plt.close(fig)


# ======================================================== figure 4: slab + capacitor
def capacitor_panel(ax):
    """Series vs parallel capacitor picture of normal vs tangential E.

    Arrow lengths are to scale: the normal field in the silica is exactly
    (n_Si/n_ox)^2 times the field in the silicon.
    """
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.0)
    ratio = (n_si / n_ox) ** 2
    sq = "\u00b2"
    for x0 in (0.3, 5.5):
        ax.add_patch(Rectangle((x0, 0.6), 4.2, 2.2, color=C_SI))
        ax.add_patch(Rectangle((x0, 2.8), 4.2, 2.4, color=C_OX, ec="#cfd4da"))
        ax.text(x0 + 4.1, 4.85, f"silica  n{sq} = {n_ox**2:.2f}", fontsize=8.3, ha="right",
                color=C_MUTED)
        ax.text(x0 + 4.1, 0.75, f"silicon  n{sq} = {n_si**2:.1f}", fontsize=8.3, ha="right",
                color=C_MUTED)
        ax.plot([x0, x0 + 4.2], [2.8, 2.8], color=C_INK, lw=1.2)
    for xx in (1.0, 1.9, 2.8):
        ax.add_patch(FancyArrowPatch((xx, 2.30), (xx, 2.30 + 0.34), arrowstyle="-|>",
                                     mutation_scale=10, lw=2.2, color=C_TM))
        ax.add_patch(FancyArrowPatch((xx, 2.88), (xx, 2.88 + 0.34 * ratio),
                                     arrowstyle="-|>", mutation_scale=10, lw=2.2, color=C_TM))
    ax.text(2.4, 5.45, "E NORMAL to the face\n(quasi-TM at the top and bottom)", ha="center",
            fontsize=9, weight="bold", color=C_TM, va="bottom")
    ax.text(2.4, 0.05, f"D = n{sq}E is equal on both sides, so E is\nx{ratio:.2f} stronger "
            "in the silica: a series capacitor", ha="center", fontsize=8.3)
    for yy in (1.5, 2.2, 3.4, 4.1):
        ax.add_patch(FancyArrowPatch((6.0, yy), (8.6, yy), arrowstyle="-|>",
                                     mutation_scale=10, lw=2.2, color=C_TE))
    ax.text(7.6, 5.45, "E ALONG the face\n(quasi-TE at the top and bottom)", ha="center",
            fontsize=9, weight="bold", color=C_TE, va="bottom")
    ax.text(7.6, 0.05, "tangential E is equal on both sides:\nno jump, a parallel capacitor",
            ha="center", fontsize=8.3)


say("figure 4: the slab picture and the capacitor analogy")
fig, axs = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
xx = np.linspace(-0.6, 0.6, 2401)
ax = axs[0]
ax.axvspan(-H / 2, H / 2, color=C_SI, alpha=0.55, lw=0)
ax.plot(xx, tetm.slab_fields(LAM, H, n_si, n_ox, slab["TE"], xx, "TE"), color=C_TE, lw=2.3,
        label=f"TE slab: E along the faces,  n$_{{eff}}$ = {slab['TE']:.3f}")
tm_line = tetm.slab_fields(LAM, H, n_si, n_ox, slab["TM"], xx, "TM")
ax.plot(xx, tm_line, color=C_TM, lw=2.3,
        label=f"TM slab: E across the faces,  n$_{{eff}}$ = {slab['TM']:.3f}")
k0 = 2 * math.pi / LAM
dec = {p: 1000 / (k0 * math.sqrt(slab[p] ** 2 - n_ox**2)) for p in slab}
R["slab_decay_nm"] = dec
ax.annotate(f"TM tail decays in {dec['TM']:.0f} nm", xy=(0.25, tm_line[np.argmin(np.abs(xx - 0.25))]),
            xytext=(0.30, 0.95), fontsize=9, color=C_TM,
            arrowprops=dict(arrowstyle="->", color=C_TM))
ax.annotate(f"TE tail decays in {dec['TE']:.0f} nm", xy=(0.19, 0.14), xytext=(0.30, 0.55),
            fontsize=9, color=C_TE, arrowprops=dict(arrowstyle="->", color=C_TE))
ax.set_xlabel("x, across the 220 nm slab (um)")
ax.set_ylabel("E across the slab (1 at the centre)")
ax.set_title("Exact 1-D slab, 220 nm silicon in silica: the TM field jumps x"
             f"{(n_si/n_ox)**2:.2f} at each face\nand spreads much further into the silica",
             fontsize=10, loc="left")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.03, 1.55)
capacitor_panel(axs[1])
axs[1].set_title("Why: a field pointing across a boundary is a series capacitor",
                 fontsize=10, loc="left")
fig.savefig(OUT / "4_slab_and_capacitor.png")
plt.close(fig)

# ======================================================== figure 5: mode map
say("figure 5: mode map, n_eff versus width and versus thickness (about a minute)")


def sweep(param, values):
    pts = []
    for v in values:
        w, h = (v, H) if param == "w" else (W, v)
        for m in tetm.solve(LAM, w, h, num_modes=4, res=0.03, pad=1.0):
            # modes within 0.05 of cutoff have tails longer than the 1 um box can
            # hold, so their n_eff is not trustworthy; leave them out
            if m.n_eff > n_ox + 0.05:
                pts.append((v, m.n_eff, m.te_fraction))
    return np.array(pts)


SW = sweep("w", np.arange(0.20, 0.901, 0.025))
SH = sweep("h", np.arange(0.12, 0.601, 0.02))
R["width_sweep"] = SW.tolist()
R["thickness_sweep"] = SH.tolist()

fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"wspace": 0.32})
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("tetm", [C_TM, "#bdbab3", C_TE])
for ax, S, lab, ref, other in ((axs[0], SW, "strip width (nm)", W, H),
                               (axs[1], SH, "silicon thickness (nm)", H, W)):
    sc = ax.scatter(S[:, 0] * 1000, S[:, 1], c=S[:, 2], cmap=cmap, vmin=0, vmax=1,
                    s=28, edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(n_ox, color=C_MUTED, ls=":", lw=1)
    ax.text(S[:, 0].max() * 1000, n_ox + 0.015, f"silica index {n_ox:.3f}: cutoff",
            fontsize=8, color=C_MUTED, ha="right")
    ax.axvline(ref * 1000, color=C_INK, ls="--", lw=1)
    ax.axvline(other * 1000, color=C_MUTED, ls=":", lw=1)
    ax.set_xlabel(lab)
    ax.set_ylabel("n$_{eff}$")
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("TE fraction (1 = E in the chip plane)", fontsize=8)
axs[0].text(W * 1000 + 8, 2.43, "500 nm\nreference", fontsize=8.5)
axs[0].text(H * 1000 + 10, 2.80, "square\n220 x 220:\nTE = TM", fontsize=8.5, color=C_MUTED)
axs[1].text(H * 1000 + 8, 1.75, "220 nm\nreference", fontsize=8.5)
axs[1].text(W * 1000 + 8, 1.85, "square\n500 x 500:\nTE = TM", fontsize=8.5, color=C_MUTED)
axs[0].set_title("Widen the strip at 220 nm thickness: TE0 pulls away from TM0.\n"
                 "At 500 nm it also guides TE1; TE1 and TM0 cross freely because the cladding is symmetric",
                 fontsize=9.5, loc="left")
axs[1].set_title("Thicken the silicon at 500 nm width: TM catches up and passes TE\n"
                 "once the strip is taller than it is wide (the roles swap)",
                 fontsize=9.5, loc="left")
fig.suptitle("Which polarization wins is decided by the aspect ratio: the field prefers to lie "
             "along the long faces", fontsize=12, y=1.03)
fig.savefig(OUT / "5_mode_map.png")
plt.close(fig)

# ======================================================== figure 6: capstone numbers
say("figure 6: what it means for the ring (group index needs 6 more solves)")
ngTE = tetm.group_index(LAM, W, H, "TE", res=0.02)[1]
ngTM = tetm.group_index(LAM, W, H, "TM", res=0.02)[1]
for key, ng in (("TE", ngTE), ("TM", ngTM)):
    R[key]["n_g"] = ng
    R[key]["fsr_nm_39p6um_ring"] = LAM**2 / (ng * RING_L) * 1000
    R[key]["dlambda_dT_pm_per_K"] = LAM / ng * R[key]["dneff_dnsi"] * 1.86e-4 * 1e6
lamTE = TE.n_eff * RING_L / round(TE.n_eff * RING_L / LAM)
R["ring_resonance_split_nm"] = None

fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1.3, 0.55, 0.55, 1.05], "wspace": 0.35})
ax = axs[0]
labels = ["n$_{eff}$", "n$_g$", "∂n$_{eff}$/∂n$_{Si}$", "power in Si"]
te_v = [R["TE"]["n_eff"], ngTE, R["TE"]["dneff_dnsi"], R["TE"]["power_fraction_in_core"]]
tm_v = [R["TM"]["n_eff"], ngTM, R["TM"]["dneff_dnsi"], R["TM"]["power_fraction_in_core"]]
xb = np.arange(len(labels))
ax.bar(xb - 0.19, te_v, 0.36, color=C_TE, label="quasi-TE0")
ax.bar(xb + 0.19, tm_v, 0.36, color=C_TM, label="quasi-TM0")
for i, (a, b) in enumerate(zip(te_v, tm_v)):
    ax.text(i - 0.19, a + 0.05, f"{a:.2f}", ha="center", fontsize=8.5, color=C_TE)
    ax.text(i + 0.19, b + 0.05, f"{b:.2f}", ha="center", fontsize=8.5, color=C_TM)
ax.set_xticks(xb, labels)
ax.set_title("The two modes are different waveguides in all but name", fontsize=10, loc="left")
ax.legend(fontsize=9)
ax.grid(axis="x")

for ax, key, unit, ttl in ((axs[1], "fsr_nm_39p6um_ring", "nm",
                            "FSR of the same 39.6 um ring"),
                           (axs[2], "dlambda_dT_pm_per_K", "pm/K",
                            "resonance shift per kelvin")):
    a_, b_ = R["TE"][key], R["TM"][key]
    ax.bar([0], [a_], 0.6, color=C_TE)
    ax.bar([1], [b_], 0.6, color=C_TM)
    ax.text(0, a_ * 1.02, f"{a_:.1f}", ha="center", fontsize=9.5, color=C_TE)
    ax.text(1, b_ * 1.02, f"{b_:.1f}", ha="center", fontsize=9.5, color=C_TM)
    ax.set_xticks([0, 1], ["TE0", "TM0"])
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(a_, b_) * 1.18)
    ax.set_title(ttl, fontsize=10, loc="left")
    ax.grid(axis="x")
xb = np.arange(2)

ax = axs[3]
names = ["|E|" + "²" + " just outside\nthe sidewall", "|E|" + "²" + " just above\nthe top face"]
tv = [R["TE"]["E2_at_sidewall_over_peak"], R["TE"]["E2_at_topface_over_peak"]]
mv = [R["TM"]["E2_at_sidewall_over_peak"], R["TM"]["E2_at_topface_over_peak"]]
ax.bar(xb - 0.19, tv, 0.36, color=C_TE)
ax.bar(xb + 0.19, mv, 0.36, color=C_TM)
for i, (a, b) in enumerate(zip(tv, mv)):
    ax.text(i - 0.19, a + 0.02, f"{a:.2f}", ha="center", fontsize=9, color=C_TE)
    ax.text(i + 0.19, b + 0.02, f"{b:.2f}", ha="center", fontsize=9, color=C_TM)
ax.set_xticks(xb, names)
ax.set_ylabel("relative to the peak |E|" + "²")
ax.set_title("Where each mode is strong: TE at the rough etched\nsidewalls, TM at the smooth top and bottom",
             fontsize=10, loc="left")
ax.grid(axis="x")
fig.suptitle("Why the platform is designed for quasi-TE, in numbers (500 x 220 nm, 1310 nm)",
             fontsize=12, y=1.04)
fig.savefig(OUT / "6_capstone.png")
plt.close(fig)

# ======================================================== the poster
say("poster: one page")
fig = plt.figure(figsize=(17, 15.5))
gsp = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], hspace=0.42, wspace=0.45,
                       top=0.915, bottom=0.04)
a1 = fig.add_subplot(gsp[0, 0:2])
a2 = fig.add_subplot(gsp[0, 2:4])
for ax, key in ((a1, "TE"), (a2, "TM")):
    cross_section(ax, key)
    r = R[key]
    ax.set_title(("quasi-TE0: E across the width, in the chip plane" if key == "TE"
                  else "quasi-TM0: E across the thickness, out of the chip")
                 + f"\nn$_{{eff}}$ = {r['n_eff']:.3f},  n$_g$ = {r['n_g']:.3f},  "
                 f"{r['power_fraction_in_core']*100:.0f} % of the power in the silicon",
                 fontsize=11, color=C_TE if key == "TE" else C_TM, loc="left", weight="bold")
b = [fig.add_subplot(gsp[1, i]) for i in range(4)]
draw_cut(b[0], "TE", 0, "TE, horizontal cut:\nEx NORMAL to the sidewalls")
draw_cut(b[1], "TE", 1, "TE, vertical cut:\nEx TANGENTIAL to top/bottom")
draw_cut(b[2], "TM", 1, "TM, vertical cut:\nEy NORMAL to top/bottom")
draw_cut(b[3], "TM", 0, "TM, horizontal cut:\nEy TANGENTIAL to the sidewalls")
b[0].set_ylabel("field (1 = peak)")
c1 = fig.add_subplot(gsp[2, 0:2])
c2 = fig.add_subplot(gsp[2, 2:4])
c1.scatter(SW[:, 0] * 1000, SW[:, 1], c=SW[:, 2], cmap=cmap, vmin=0, vmax=1, s=26,
           edgecolors="white", linewidths=0.4, zorder=3)
c1.axhline(n_ox, color=C_MUTED, ls=":", lw=1)
c1.axvline(W * 1000, color=C_INK, ls="--", lw=1)
c1.axvline(H * 1000, color=C_MUTED, ls=":", lw=1)
c1.text(W * 1000 + 8, 2.43, "500 nm", fontsize=8.5)
c1.text(H * 1000 + 10, 2.80, "square:\nTE = TM", fontsize=8.5, color=C_MUTED)
c1.set_xlabel("strip width (nm), thickness fixed at 220 nm")
c1.set_ylabel("n$_{eff}$  (blue = TE-like, orange = TM-like)")
c1.set_title("Wider than thick, the TE mode wins; in a square guide they are identical",
             fontsize=10.5, loc="left")
capacitor_panel(c2)
c2.set_title("The physics in one picture: D = n" + "²" + "E is continuous across a face, "
             "so a normal field jumps", fontsize=10.5, loc="left")
fig.suptitle("TE versus TM in a silicon waveguide  (500 x 220 nm strip in silica, 1310 nm, "
             "full-vector FEM solve)", fontsize=15, y=0.995, weight="bold")
fig.text(0.5, 0.955,
         "Quasi-TE keeps its field along the two long faces, where tangential E is continuous, so it stays in the "
         "silicon. Quasi-TM points across those faces, where E must jump\nby "
         f"{(n_si/n_ox)**2:.2f}x into the silica, so it leaks out: lower n$_{{eff}}$ "
         f"({TM.n_eff:.2f} vs {TE.n_eff:.2f}), a longer tail and a different ring. "
         "That is why the platform is built for quasi-TE.",
         ha="center", fontsize=10.5, color=C_MUTED)
fig.savefig(OUT / "te_vs_tm_poster.png")
plt.close(fig)

# ======================================================== animation
say("animation: the hybrid fields travelling along the guide")
lam_g_te = LAM / TE.n_eff
lam_g_tm = LAM / TM.n_eff
zz = np.linspace(0, 2.2, 260)
iy0, ix0 = np.argmin(np.abs(ys)), np.argmin(np.abs(xs))
sel_x = np.abs(xs) <= 0.6
sel_y = np.abs(ys) <= 0.45
te_ex, te_ez = G["TE"]["Ex"][iy0, sel_x], G["TE"]["Ez"][iy0, sel_x]     # top view, y = 0
tm_ey, tm_ez = G["TM"]["Ey"][sel_y, ix0], G["TM"]["Ez"][sel_y, ix0]     # side view, x = 0
xa, ya = xs[sel_x], ys[sel_y]
bte = 2 * math.pi * TE.n_eff / LAM
btm = 2 * math.pi * TM.n_eff / LAM


def inst(trans, longi, beta, phase):
    """Physical field Re{E(u) exp(j(phi - beta z))} on the (u, z) plane."""
    ph = np.exp(1j * (phase - beta * zz))[None, :]
    return np.real(trans[:, None] * ph), np.real(longi[:, None] * ph)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.6), gridspec_kw={"hspace": 0.5})
T0, Z0 = inst(te_ex, te_ez, bte, 0)
T1, Z1 = inst(tm_ey, tm_ez, btm, 0)
im1 = ax1.imshow(T0, extent=[zz[0], zz[-1], xa[0], xa[-1]], origin="lower", aspect="equal",
                 cmap="RdBu_r", vmin=-1, vmax=1)
im2 = ax2.imshow(T1, extent=[zz[0], zz[-1], ya[0], ya[-1]], origin="lower", aspect="equal",
                 cmap="RdBu_r", vmin=-1, vmax=1)
for ax, half in ((ax1, W / 2), (ax2, H / 2)):
    ax.axhline(half, color=C_INK, lw=1.2)
    ax.axhline(-half, color=C_INK, lw=1.2)
    ax.grid(False)
qz, qu1, qu2 = zz[::13], xa[::8], ya[::6]
QZ1, QU1 = np.meshgrid(qz, qu1)
QZ2, QU2 = np.meshgrid(qz, qu2)
QK = dict(color=C_INK, angles="xy", scale_units="xy", scale=9.0, width=0.0026,
          headwidth=3.5, headlength=4)
q1 = ax1.quiver(QZ1, QU1, Z0[::8, ::13], T0[::8, ::13], **QK)
q2 = ax2.quiver(QZ2, QU2, Z1[::6, ::13], T1[::6, ::13], **QK)
ax1.set_title(f"quasi-TE0 seen from ABOVE (the x-z plane at mid-thickness). Colour = Ex, arrows = (Ez, Ex).\n"
              f"Ez appears at the sidewalls, where Ex jumps: the field lines curve round the "
              f"crests, so the mode is not purely transverse.", fontsize=9.5, loc="left", color=C_TE)
ax2.set_title(f"quasi-TM0 seen from the SIDE (the y-z plane at mid-width). Colour = Ey, arrows = (Ez, Ey).\n"
              f"Here the jump and the Ez sit on the top and bottom faces, and the field reaches far "
              f"into the silica.", fontsize=9.5, loc="left", color=C_TM)
ax1.set_ylabel("x (um)")
ax2.set_ylabel("y (um)")
ax2.set_xlabel("z, along the waveguide (um)")
ax1.set_xlabel("z, along the waveguide (um)")
ax1.text(zz[-1] + 0.03, W / 2, "sidewall", ha="left", va="center", fontsize=8, clip_on=False)
ax1.text(zz[-1] + 0.03, -W / 2, "sidewall", ha="left", va="center", fontsize=8, clip_on=False)
ax2.text(zz[-1] + 0.03, H / 2, "top face", ha="left", va="center", fontsize=8, clip_on=False)
ax2.text(zz[-1] + 0.03, -H / 2, "bottom face", ha="left", va="center", fontsize=8, clip_on=False)
NF = 96


def animate(i):
    p = 2 * math.pi * i / NF * 2       # two optical periods over the clip
    a, b = inst(te_ex, te_ez, bte, p)
    c, d = inst(tm_ey, tm_ez, btm, p)
    im1.set_data(a)
    im2.set_data(c)
    q1.set_UVC(b[::8, ::13], a[::8, ::13])
    q2.set_UVC(d[::6, ::13], c[::6, ::13])
    return im1, im2, q1, q2


anim = FuncAnimation(fig, animate, frames=NF, interval=60)
import imageio_ffmpeg
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
anim.save(OUT / "te_tm_travelling.mp4", writer=FFMpegWriter(fps=24, bitrate=2600))
anim.save(OUT / "te_tm_travelling.gif", writer=PillowWriter(fps=16), dpi=60)
plt.close(fig)

R["runtime_s"] = time.time() - t0
(OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
print("\n" + "=" * 70)
for key in ("TE", "TM"):
    r = R[key]
    print(f"  quasi-{key}0: n_eff {r['n_eff']:.4f}  n_g {r['n_g']:.4f}  "
          f"power in Si {r['power_fraction_in_core']*100:.0f} %  "
          f"FSR {r['fsr_nm_39p6um_ring']:.2f} nm  {r['dlambda_dT_pm_per_K']:.1f} pm/K")
print(f"  slab 220 nm: TE {slab['TE']:.4f}  TM {slab['TM']:.4f}")
print(f"  done in {R['runtime_s']:.0f} s, outputs in {OUT}")
