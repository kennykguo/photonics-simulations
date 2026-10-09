"""Stage 2: the step pages.

One page = one question, one main visual (at most one supporting visual).
Text on the page is limited to numbered one-line keys for the markers and one
"so" line that carries the argument to the next page. The full explanation of
every page is in README.md, under the same page number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

import render3d as r3  # noqa: E402
from cmt import DN2, GAP, H, K0, LAM, N_OX, N_SI, W  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PAGES = OUT / "pages"
REND = OUT / "renders"
for d in (PAGES, REND):
    d.mkdir(parents=True, exist_ok=True)

A_COL, B_COL, P_COL = r3.A_COL, r3.B_COL, "#6a3d9a"
INK, GREY, NAVY = "#1d2733", "#6b7785", "#1d3d63"
plt.rcParams.update({
    "font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 12.5, "axes.titleweight": "bold",
    "axes.titlepad": 10, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#55606b",
    "xtick.color": "#3a4652", "ytick.color": "#3a4652", "axes.labelcolor": INK,
    "text.color": INK, "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

CHAIN = ["one guide", "add B", "A's tail\nreaches B", "$u_A$ no longer\nexact", "two-mode\nblend",
         "project\nonto B", "the number\n$\\kappa_c$", "power\noscillates", "supermodes\nbeat",
         "3-D Maxwell\ncheck", "gap\nsensitivity"]
NSTEP = len(CHAIN)
KEY_MAX = 135      # characters per key line; longer keys are flagged, never wrapped


def tag(ax, x, y, n, transform=None, size=12.5):
    tr = transform if transform is not None else ax.transData
    ax.annotate(str(n), (x, y), xycoords=tr, ha="center", va="center", fontsize=size,
                fontweight="bold", color="white", zorder=50,
                bbox=dict(boxstyle="circle,pad=0.28", fc=INK, ec="white", lw=1.4))


class Page:
    count = 0

    def __init__(self, step, question, part=None):
        Page.count += 1
        self.idx, self.step = Page.count, step
        self.fig = f = plt.figure(figsize=(16, 9))
        lab = f"STEP {step} OF {NSTEP}" + (f"   ·   PAGE {part[0]} OF {part[1]}" if part else "")
        f.text(0.035, 0.955, lab, fontsize=12, color=GREY, fontweight="bold", va="center")
        f.text(0.035, 0.912, question, fontsize=23, fontweight="bold", va="center")
        x0, x1, y = 0.035, 0.965, 0.842
        wb = (x1 - x0) / NSTEP
        for i, c in enumerate(CHAIN, start=1):
            cur, done = i == step, i < step
            f.patches.append(FancyBboxPatch(
                (x0 + (i - 1) * wb + 0.002, y - 0.026), wb - 0.004, 0.052,
                boxstyle="round,pad=0,rounding_size=0.006", transform=f.transFigure,
                fc=NAVY if cur else ("#dce8f5" if done else "#f3f4f6"),
                ec=NAVY if cur else ("#9fbbd9" if done else "#d5d9de")))
            f.text(x0 + (i - 0.5) * wb, y, f"{i}. " + c, fontsize=9, ha="center", va="center",
                   color="white" if cur else (INK if done else "#9aa3ad"),
                   fontweight="bold" if cur else "normal", linespacing=1.05)

    def footer(self, keys, so):
        f = self.fig
        f.patches.append(FancyBboxPatch((0.035, 0.022), 0.93, 0.142,
                                        boxstyle="round,pad=0,rounding_size=0.008",
                                        transform=f.transFigure, fc="#f6f7f9", ec="#d5d9de"))
        y = 0.135
        for i, k in enumerate(keys, start=1):
            if len(k) > KEY_MAX:
                print(f"  ! page {self.idx}: key {i} is {len(k)} chars")
            f.text(0.052, y, str(i), fontsize=11, fontweight="bold", color="white", ha="center",
                   va="center", bbox=dict(boxstyle="circle,pad=0.22", fc=INK, ec="none"))
            f.text(0.066, y, k, fontsize=13, va="center")
            y -= 0.031
        f.text(0.045, 0.040, "so  →  " + so, fontsize=13.5, va="center", color=NAVY, fontweight="bold")
        if len(keys) > 3:
            print(f"  ! page {self.idx}: {len(keys)} keys")

    def save(self, name, pdf):
        path = PAGES / f"{self.idx:02d}_step{self.step:02d}_{name}.png"
        self.fig.savefig(path, dpi=120)
        pdf.savefig(self.fig)
        plt.close(self.fig)
        return path


def trim(path, pad=10):
    img = mpimg.imread(path)
    nz = np.where((img[..., :3] < 0.985).any(axis=2))
    r0, r1, c0, c1 = nz[0].min(), nz[0].max(), nz[1].min(), nz[1].max()
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    return img[r0:r1 + pad, c0:c1 + pad], (r0, c0)


def show_render(ax, path, anchors=None):
    img, (r0, c0) = trim(path)
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    return {k: (x - c0, y - r0) for k, (x, y) in (anchors or {}).items()}


def core_rects(ax, xa, xb, lw=1.8, only=None, ls_b="-"):
    for xc, col, name in ((xa, A_COL, "A"), (xb, B_COL, "B")):
        if only and name not in only:
            continue
        ax.add_patch(Rectangle((xc - W / 2, -H / 2), W, H, fill=False, ec=col, lw=lw,
                               ls=ls_b if name == "B" else "-", zorder=20))


def shade_cores(ax, xa, xb, alpha=0.10):
    for xc, col in ((xa, A_COL), (xb, B_COL)):
        ax.axvspan(xc - W / 2, xc + W / 2, color=col, alpha=alpha, lw=0)


def local_rate(x, f, win=0.06, step=0.02):
    """-d ln|f|/dx from straight-line fits over sliding windows (smooths FEM noise)."""
    cs, rs = [], []
    for c in np.arange(x[0] + win / 2, x[-1] - win / 2, step):
        s = (x > c - win / 2) & (x < c + win / 2)
        rs.append(-np.polyfit(x[s], np.log(np.abs(f[s])), 1)[0])
        cs.append(c)
    return np.array(cs), np.array(rs)


def load_fdtd():
    """All FDTD runs present, finest grid first: [(npz dict, json dict), ...]."""
    runs = []
    for js in sorted((OUT / "fdtd").glob("fdtd3d_res*.json")):
        npz = js.with_suffix(".npz")
        if npz.exists():
            runs.append((dict(np.load(npz)), json.loads(js.read_text())))
    return sorted(runs, key=lambda r: -r[1]["resolution_px_per_um"]) or None


def fit_kappa(F, guess):
    z, PA, PB = F["z"], F["PA"], F["PB"]
    sel = z >= 0
    fB = (PB / (PA + PB))[sel]
    f = lambda z_, k_, z0, a_, c_: a_ * np.sin(k_ * (z_ - z0)) ** 2 + c_  # noqa: E731
    popt, _ = curve_fit(f, z[sel], fB, p0=[guess, 0.0, 1.0, 0.0], maxfev=20000)
    return abs(popt[0]), float(fB.max())


# --------------------------------------------------------------------------


def build_all(D, R, fd=None):
    Page.count = 0
    for old in PAGES.glob("*.png"):      # page numbers shift when pages are added; start clean
        old.unlink()
    xs, ys, xl = D["xs"], D["ys"], D["xl"]
    xa, xb = float(D["xa"]), float(D["xb"])
    X, Y = np.meshgrid(xs, ys)
    iy0 = int(np.argmin(np.abs(ys)))
    S, V = R["scalar"], R["vector"]
    coresAB = [(xa, W, H, A_COL), (xb, W, H, B_COL)]
    inB = (np.abs(X - xb) <= W / 2) & (np.abs(Y) <= H / 2)
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    pdf = PdfPages(OUT / "coupled_modes_walkthrough.pdf")
    paths = []

    # ================================================================ step 1
    nA = S["n_A"]
    p = Page(1, "What does a single waveguide carry?")
    zs = 0.45
    anc = r3.height_surface(str(REND / "s1_uA.png"), X, Y, D["uA"], [(xa, W, H, A_COL)],
                            cmap=r3.CMAP_A, zscale=zs,
                            anchors={"peak": (xa, 0, zs), "skirt": (xa + 0.45, 0.0, 0.01)})
    ax = p.fig.add_axes([0.03, 0.19, 0.60, 0.60])
    a = show_render(ax, REND / "s1_uA.png", anc)
    ax.set_title("mode shape $u_A(x, y)$ of guide A, drawn as height", loc="left")
    tag(ax, a["peak"][0] + 55, a["peak"][1], 1)
    tag(ax, a["skirt"][0] + 70, a["skirt"][1] - 40, 2)
    ax2 = p.fig.add_axes([0.68, 0.30, 0.29, 0.42])
    u = D["uA"] / D["uA"].max()
    ax2.imshow(u, extent=ext, origin="lower", cmap=r3.CMAP_A, vmin=0, vmax=1, aspect="equal")
    cs = ax2.contour(X, Y, u, levels=[0.01, 0.1], colors=[GREY], linewidths=1)
    ax2.clabel(cs, fmt={0.01: "1 %", 0.1: "10 %"}, fontsize=10)
    core_rects(ax2, xa, xb, only="A")
    ax2.set_xlim(-1.3, 0.55)
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("y (µm)")
    ax2.set_title("the same mode, end-on", loc="left")
    p.footer([f"The mode shape: how strong the field is across the guide. Solved from the scalar wave equation; "
              f"$n_{{eff}}$ = {nA:.3f}.",
              "The evanescent tail: the field does not stop at the silicon sidewall (contours: 10 % and 1 % of peak)."],
             f"one guide carries one fixed shape that travels as $e^{{-j\\beta z}}$, β = {K0*nA:.2f} rad/µm.  "
             "Nothing is coupled yet.")
    paths.append(p.save("one_guide", pdf))

    # ================================================================ step 2
    p = Page(2, "What changes physically when guide B is placed next to A?")
    maps = [np.full_like(X, N_OX**2),
            np.where((np.abs(X - xa) <= W / 2) & (np.abs(Y) <= H / 2), DN2, 0.0),
            np.where(inB, DN2, 0.0)]
    maps.append(maps[0] + maps[1] + maps[2])
    titles = ["$n_{clad}^2$", "$\\Delta_A$", "$\\Delta_B$  (new)", "$n^2$ of the coupler"]
    wm, gapm, x0 = 0.19, 0.055, 0.045
    for i, (m, t) in enumerate(zip(maps, titles)):
        axm = p.fig.add_axes([x0 + i * (wm + gapm), 0.33, wm, 0.36])
        axm.imshow(m, extent=ext, origin="lower", cmap="Greys", vmin=0, vmax=N_SI**2 * 1.05,
                   aspect="equal")
        core_rects(axm, xa, xb, lw=1.2)
        axm.set_title(t, loc="center", color=B_COL if i == 2 else INK, fontsize=15)
        axm.set_xticks([-1, 0, 1])
        axm.set_yticks([-0.5, 0, 0.5])
        axm.set_xlabel("x (µm)", fontsize=11)
        if i < 3:
            p.fig.text(x0 + (i + 1) * (wm + gapm) - gapm / 2, 0.51, "=" if i == 2 else "+",
                       fontsize=30, ha="center", va="center", fontweight="bold")
        if i == 1:
            tag(axm, -1.05, 0.4, 1)
        if i == 2:
            tag(axm, -1.05, 0.4, 2)
    p.footer([f"A mask Δ is $n_{{Si}}^2 - n_{{ox}}^2$ = {DN2:.2f} inside its core and 0 everywhere else.  "
              "$u_A$ was solved with only the first two terms.",
              "Placing B adds exactly one term, $\\Delta_B$, which is non-zero only inside B's rectangle."],
             "the wave equation is unchanged everywhere except inside B.  Does A's field reach that far?")
    paths.append(p.save("index_masks", pdf))

    # ================================================================ step 3
    p = Page(3, "Does A's field actually reach B's silicon?")
    u = np.abs(D["uA"]) / np.abs(D["uA"]).max()
    ax = p.fig.add_axes([0.05, 0.22, 0.40, 0.52])
    im = ax.imshow(u, extent=ext, origin="lower", cmap=r3.CMAP_A, norm=LogNorm(1e-4, 1), aspect="equal")
    cs = ax.contour(X, Y, u, levels=[1e-4, 1e-3, 1e-2, 1e-1], colors="#33414f", linewidths=0.8)
    ax.clabel(cs, fmt={1e-4: "0.01 %", 1e-3: "0.1 %", 1e-2: "1 %", 1e-1: "10 %"}, fontsize=9.5)
    core_rects(ax, xa, xb, ls_b="--")
    ax.text(xa, 0.0, "A", color="white", ha="center", va="center", fontweight="bold", fontsize=14)
    ax.text(xb, H / 2 + 0.05, "B", color=B_COL, ha="center", fontweight="bold", fontsize=14)
    p.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("$|u_A|$ / peak", labelpad=-2)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("guide A's mode on a log colour scale", loc="left")
    tag(ax, -1.12, 0.42, 1)
    ul = np.abs(D["uA_line"]) / np.abs(D["uA_line"]).max()
    face = xb - W / 2
    val = lambda x_: float(np.interp(x_, xl, ul))  # noqa: E731
    ax3 = p.fig.add_axes([0.62, 0.24, 0.34, 0.48])
    ax3.semilogy(xl, ul, color=A_COL, lw=2.4)
    g = S["gamma_tail_fit"]
    shade_cores(ax3, xa, xb, alpha=0.12)
    ax3.text(xa, 1.3e-5, "A", color=A_COL, ha="center", fontweight="bold", fontsize=13)
    ax3.text(xb, 1.3e-5, "B", color=B_COL, ha="center", fontweight="bold", fontsize=13)
    ax3.plot([face], [val(face)], "o", color=B_COL, ms=7)
    ax3.annotate(f"{val(face)*100:.1f} % of peak\nat B's near face", (face, val(face)), xytext=(0.62, 0.15),
                 fontsize=11.5, color=B_COL, arrowprops=dict(arrowstyle="->", color=B_COL))
    ax3.set_ylim(1e-5, 2)
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_xlabel("x (µm), cut through the middle (y = 0)")
    ax3.set_ylabel("$|u_A|$ / peak")
    ax3.set_title("the same field along one line", loc="left")
    tag(ax3, -0.05, 3e-3, 2)
    p.footer(["Every contour is 10x weaker than the one inside it.  The faint tail plainly overlaps B's rectangle (dashed).",
              f"A straight line on a log axis is an exponential: between the cores the field falls by e every "
              f"{1e3/g:.0f} nm."],
             f"part of A's field sits in B's silicon ({S['frac_A_in_B']*100:.3f} % of $|u_A|^2$), "
             "a region A's own equation treated as silica.")
    paths.append(p.save("tail_reaches_B", pdf))

    # ================================================================ step 4
    p = Page(4, "Where exactly does $u_A$ stop being a solution?", (1, 2))
    zs = 0.42
    r3.height_surface(str(REND / "s4_uA.png"), X, Y, D["uA"], coresAB, cmap=r3.CMAP_A, zscale=zs)
    r3.height_surface(str(REND / "s4_mask.png"), X, Y, np.where(inB, 1.0, 0.0), coresAB,
                      cmap=r3.CMAP_B, zscale=0.18)
    prod = D["uA"] * inB
    mag = np.abs(D["uA"]).max() / np.abs(prod).max()
    anc_c = r3.height_surface(str(REND / "s4_product.png"), X, Y, prod, coresAB, cmap=r3.CMAP_P,
                              zscale=zs, anchors={"peak": (xb - W / 2, 0, zs)})
    wp, gp = 0.285, 0.04
    axs = [p.fig.add_axes([0.035 + i * (wp + gp), 0.27, wp, 0.42]) for i in range(3)]
    show_render(axs[0], REND / "s4_uA.png")
    axs[0].set_title("$u_A$  (old solution)", loc="center", fontsize=15)
    show_render(axs[1], REND / "s4_mask.png")
    axs[1].set_title("$\\Delta_B$  (new term)", loc="center", fontsize=15)
    cc = show_render(axs[2], REND / "s4_product.png", anc_c)
    axs[2].set_title("$\\Delta_B \\cdot u_A$", loc="center", fontsize=15)
    for i, s_ in ((0, "×"), (1, "=")):
        p.fig.text(0.035 + (i + 1) * (wp + gp) - gp / 2, 0.48, s_, fontsize=34, ha="center",
                   va="center", fontweight="bold")
    tag(axs[2], cc["peak"][0] - 80, cc["peak"][1] + 30, 1)
    p.footer([f"Multiplying by the mask keeps only the part of A's field inside B's silicon.  It is tiny: drawn "
              f"{mag:.0f}x taller than the left plot."],
             "if $u_A$ fails anywhere, it can only be here.  Next page: check that on the computer.")
    paths.append(p.save("residual_product", pdf))

    p = Page(4, "Where exactly does $u_A$ stop being a solution?", (2, 2))
    res = D["residual"]
    ax = p.fig.add_axes([0.06, 0.22, 0.56, 0.52])
    im = ax.imshow(res / np.abs(res).max(), extent=ext, origin="lower", cmap="PuOr_r",
                   norm=TwoSlopeNorm(0, -1, 1), aspect="equal")
    core_rects(ax, xa, xb)
    ax.text(xa, H / 2 + 0.05, "A", color=A_COL, ha="center", fontweight="bold", fontsize=14)
    ax.text(xb, H / 2 + 0.05, "B", color=B_COL, ha="center", fontweight="bold", fontsize=14)
    p.fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label("left-over / its maximum")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("computed: $\\nabla_t^2 u_A + (k_0^2 n^2_{pair} - \\beta_A^2)\\,u_A$", loc="left")
    tag(ax, xb + 0.55, 0.0, 1)
    axn = p.fig.add_axes([0.76, 0.30, 0.22, 0.40])
    axn.set_axis_off()
    axn.text(0, 0.95, "measured", fontweight="bold", fontsize=14, va="top", color=GREY)
    axn.text(0, 0.75, "inside B", fontsize=13, va="top")
    axn.text(0, 0.63, f"left-over / $k_0^2\\Delta_B u_A$ = {S['residual_over_prediction_in_B_median']:.5f}",
             fontsize=13, va="top")
    axn.text(0, 0.40, "everywhere else", fontsize=13, va="top")
    axn.text(0, 0.28, f"< {S['residual_outside_B_max_rel']*100:.2f} % of its peak", fontsize=13, va="top")
    tag(axn, -0.06, 0.60, 2, transform=axn.transAxes)
    p.footer(["Not drawn from a formula: $u_A$ was put into the two-guide equation and what does not cancel was plotted.",
              "The left-over is exactly $k_0^2\\Delta_B u_A$: the 'source term' of the notes, located only in B."],
             "a left-over means the field cannot stay as it is.  It is forced to change.  Into what?")
    paths.append(p.save("residual_computed", pdf))

    # ================================================================ step 5
    uP, uM = D["uP"][iy0], D["uM"][iy0]
    bp, bm = K0 * S["n_plus"], K0 * S["n_minus"]
    cp, cm = S["c_plus"], S["c_minus"]
    Lcs = S["L_c_exact"]
    uAl, uBl = D["uA"][iy0], D["uB0"][iy0]
    nrm = np.abs(uAl).max()
    p = Page(5, "If the field must change, what does it change into?", (1, 2))
    ax = p.fig.add_axes([0.10, 0.235, 0.62, 0.515])
    zsnap = [0, Lcs / 4, Lcs / 2, 3 * Lcs / 4, Lcs]
    for k_, z_ in enumerate(zsnap):
        fld = cp * uP * np.exp(-1j * bp * z_) + cm * uM * np.exp(-1j * bm * z_)
        off = -1.3 * k_
        ax.fill_between(xs, off, off + np.abs(fld) / nrm, color="#d6dce3")
        ax.plot(xs, off + np.abs(fld) / nrm, color=INK, lw=1.6)
        aw, bw = abs(math.cos(S["kappa_exact"] * z_)), abs(math.sin(S["kappa_exact"] * z_))
        ax.plot(xs, off + aw * np.abs(uAl) / nrm, color=A_COL, lw=1.8, ls="--")
        ax.plot(xs, off + bw * np.abs(uBl) / nrm, color=B_COL, lw=1.8, ls="--")
        ax.text(0.80, off + 0.45, f"z = {z_:4.1f} µm", fontsize=12, va="center", family="monospace")
        ax.text(0.80, off + 0.12, f"A = {aw:.2f}   B = {bw:.2f}", fontsize=11.5, va="center",
                color=GREY, family="monospace")
    shade_cores(ax, xa, xb, alpha=0.07)
    ax.set_xlim(-1.1, 1.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("x (µm), across both guides")
    ax.set_title("the exact field at five positions along the coupler", loc="left")
    ax.plot([], [], color=INK, lw=6, alpha=0.25, label="exact |E| (solved)")
    ax.plot([], [], color=A_COL, ls="--", label="A $\\cdot u_A$")
    ax.plot([], [], color=B_COL, ls="--", label="B $\\cdot u_B$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    tag(ax, -1.0, -0.2, 1)
    tag(ax, -1.0, -2.8, 2)
    p.footer(["Grey: the exact field of the two-guide structure.  Dashed: guide A's shape and guide B's shape, scaled by A and B.",
              "They match at every position: the shapes stay put, only the two weights A and B change."],
             "the field is $[A(z)\\,u_A + B(z)\\,u_B]\\,e^{-j\\beta z}$: two numbers per position describe it.")
    paths.append(p.save("two_mode_blend", pdf))

    p = Page(5, "If the field must change, what does it change into?", (2, 2))
    z = np.linspace(0, Lcs, 6000)
    ixa, ixb = int(np.argmin(np.abs(xs - xa))), int(np.argmin(np.abs(xs - xb)))
    fa = cp * uP[ixa] * np.exp(-1j * bp * z) + cm * uM[ixa] * np.exp(-1j * bm * z)
    fb = cp * uP[ixb] * np.exp(-1j * bp * z) + cm * uM[ixb] * np.exp(-1j * bm * z)
    ax = p.fig.add_axes([0.07, 0.47, 0.72, 0.27])
    ax.plot(z, np.real(fa) / nrm, color=A_COL, lw=0.45, alpha=0.85)
    ax.plot(z, np.abs(fa) / nrm, color=A_COL, lw=2.6, label="guide A: envelope |A(z)|")
    ax.plot(z, np.real(fb) / nrm - 2.5, color=B_COL, lw=0.45, alpha=0.85)
    ax.plot(z, np.abs(fb) / nrm - 2.5, color=B_COL, lw=2.6, label="guide B: envelope |B(z)|")
    ax.add_patch(Rectangle((0, -1.15), 2.0, 2.3, fill=False, ec=INK, lw=1, ls=":"))
    ax.set_yticks([0, -2.5])
    ax.set_yticklabels(["A", "B"], fontsize=14, fontweight="bold")
    ax.set_xlim(0, Lcs)
    ax.set_xlabel("z (µm)")
    ax.set_title("field at each core centre along the guide: fast oscillation under a slow envelope", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    tag(ax, Lcs * 0.45, 1.2, 1)
    axi = p.fig.add_axes([0.07, 0.245, 0.40, 0.15])
    zz = z < 2.0
    axi.plot(z[zz], np.real(fa[zz]) / nrm, color=A_COL, lw=1.8)
    axi.plot(z[zz], np.abs(fa[zz]) / nrm, color=A_COL, lw=2.6, ls="--")
    lp = LAM / ((S["n_plus"] + S["n_minus"]) / 2)
    axi.annotate("", (0.25 + lp, 1.25), (0.25, 1.25), arrowprops=dict(arrowstyle="<->", lw=1.2))
    axi.text(0.25 + lp / 2, 1.42, f"{lp*1e3:.0f} nm", ha="center", fontsize=11)
    axi.set_ylim(-1.2, 1.75)
    axi.set_xlabel("z (µm), the dotted box above")
    axi.set_title("zoom", loc="left")
    tag(axi, 1.85, 1.2, 2)
    p.footer([f"The envelopes change over {Lcs:.0f} µm.",
              f"The carrier repeats every {lp*1e3:.0f} nm, {Lcs/lp:.0f}x faster."],
             f"A and B vary slowly, so $|A''| \\ll |2\\beta A'|$ (ratio κ/2β = {S['kappa_exact']/(2*K0*S['n_A']):.4f}): "
             "drop $A''$, keep first-order equations in A and B.")
    paths.append(p.save("two_length_scales", pdf))

    # ================================================================ step 6
    p = Page(6, "How much of A's left-over can drive B's mode?", (1, 2))
    integrand = D["uB0"] * D["uA"] * inB
    r3.height_surface(str(REND / "s6_uB.png"), X, Y, D["uB0"], coresAB, cmap=r3.CMAP_B, zscale=0.42)
    anc_i = r3.height_surface(str(REND / "s6_integrand.png"), X, Y, integrand, coresAB, cmap=r3.CMAP_P,
                              zscale=0.42, anchors={"peak": (xb - W / 2, 0, 0.42)})
    ax = p.fig.add_axes([0.03, 0.25, 0.44, 0.46])
    show_render(ax, REND / "s6_uB.png")
    ax.set_title("B's own mode shape $u_B$", loc="center", fontsize=15)
    tag(ax, 120, 80, 1)
    ax = p.fig.add_axes([0.53, 0.25, 0.44, 0.46])
    ci = show_render(ax, REND / "s6_integrand.png", anc_i)
    ax.set_title("$u_B \\times \\Delta_B u_A$   (point by point)", loc="center", fontsize=15)
    tag(ax, ci["peak"][0] - 80, ci["peak"][1] + 30, 2)
    p.fig.text(0.5, 0.48, "×", fontsize=34, ha="center", va="center", fontweight="bold")
    p.fig.text(0.5, 0.40, "left-over\n$\\Delta_B u_A$", fontsize=12, ha="center", va="top", color=P_COL)
    p.footer(["Projection asks 'how much of the left-over looks like B's mode?', like a dot product $\\hat y \\cdot \\mathbf{v}$.",
              "Multiply point by point and add up: the volume under this surface is the overlap integral, one number."],
             "$B' = -j\\kappa_c A$, with $\\kappa_c = \\frac{k_0^2}{2\\beta}\\,\\iint \\Delta_B u_B u_A\\,dA \\,/ \\iint u_B^2\\,dA$.")
    paths.append(p.save("projection", pdf))

    k0te = V["kappa_overlap"]
    rows = []
    for lab, nmode in (("TE0", V["n_B_TE0"]), ("TE1", V["n_B_TE1"]), ("TM0", V["n_B_TM0"])):
        kap = k0te * V[f"proj_{lab}"] / V["proj_TE0"]
        delta = abs(K0 * (V["n_A"] - nmode)) / 2
        Fmax = 1.0 if lab == "TE0" else kap**2 / (kap**2 + delta**2)
        rows.append((lab, V[f"proj_{lab}"] / V["proj_TE0"], nmode, Fmax))
    R["projection_rows"] = rows
    p = Page(6, "How much of A's left-over can drive B's mode?", (2, 2))
    names = [f"B's {r_[0]}\n$n_{{eff}}$ = {r_[2]:.2f}" for r_ in rows]
    colsb = [B_COL, "#e3a27f", "#bbbbbb"]
    ax = p.fig.add_axes([0.12, 0.25, 0.33, 0.46])
    ax.barh(names, [r_[1] for r_ in rows], color=colsb, height=0.6)
    for i, r_ in enumerate(rows):
        ax.text(r_[1] + 0.03, i, f"{r_[1]:.2f}" if r_[1] > 0.01 else f"{r_[1]:.0e}", va="center", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("overlap with A's left-over (TE0 = 1)")
    ax.set_title("spatial overlap", loc="left")
    tag(ax, 1.45, 2.0, 1)
    ax = p.fig.add_axes([0.60, 0.25, 0.33, 0.46])
    ax.barh(names, [r_[3] for r_ in rows], color=colsb, height=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-12, 5)
    for i, r_ in enumerate(rows):
        ax.text(r_[3] * 1.8, i, "100 %" if r_[3] > 0.995 else f"{r_[3]*100:.2g} %", va="center", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("most power that can ever build up in that mode")
    ax.set_title("overlap and phase matching", loc="left")
    tag(ax, 3e-11, 1.0, 2)
    p.footer([f"A's lopsided tail overlaps B's TE1 even better than TE0 ({rows[1][1]:.2f}x).  TM0 (wrong field direction) hardly at all.",
              f"But TE1 travels at $n_{{eff}}$ = {rows[1][2]:.2f}, not {V['n_A']:.2f}: its drive slips out of step every "
              f"{LAM/abs(V['n_A']-rows[1][2]):.1f} µm and cancels.  At most {rows[1][3]*100:.2g} % builds up."],
             "coupling needs both spatial overlap and equal β (phase matching).  Identical guides have both.")
    paths.append(p.save("phase_matching", pdf))

    # ================================================================ step 7
    p = Page(7, "What is $\\kappa_c$, and can the formula be trusted?", (1, 2))
    ax = p.fig.add_axes([0.12, 0.24, 0.62, 0.50])
    groups = ["notes' scalar model", "full Maxwell (the real strip)"]
    form = [S["kappa_overlap"], V["kappa_overlap"]]
    exact = [S["kappa_exact"], V["kappa_exact"]]
    xg = np.arange(2)
    b1 = ax.bar(xg - 0.19, form, 0.36, color="#9fbbd9", label="overlap formula (steps 4 to 6)")
    b2 = ax.bar(xg + 0.19, exact, 0.36, color=NAVY, label="exact: both guides solved as one structure")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0012, f"{b.get_height():.4f}",
                    ha="center", fontsize=12)
    for i in range(2):
        ax.text(xg[i], exact[i] + 0.0075, f"agree to {form[i]/exact[i]*100:.1f} %", ha="center",
                color=NAVY, fontsize=13, fontweight="bold")
    ax.set_xticks(xg)
    ax.set_xticklabels([f"{g_}\n$L_c$ = {math.pi/2/e_:.1f} µm" for g_, e_ in zip(groups, exact)], fontsize=12.5)
    ax.set_ylabel("$\\kappa_c$ (rad/µm)")
    ax.set_ylim(0, 0.075)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    tag(ax, -0.45, 0.064, 1)
    tag(ax, 0.55, 0.036, 2)
    p.footer([f"Inside its own (scalar) model the notes' formula is right: {S['kappa_overlap']:.5f} against the exact "
              f"{S['kappa_exact']:.5f} rad/µm.",
              f"For the real strip the vector version of the same formula is right too, but the number is "
              f"{S['kappa_exact']/V['kappa_exact']:.1f}x smaller."],
             f"$\\kappa_c$ = {V['kappa_exact']:.4f} rad/µm for this coupler.  Why is the scalar model so far off?")
    paths.append(p.save("kappa", pdf))

    dA_ = (xs[1] - xs[0]) * (ys[1] - ys[0])
    ntA = math.sqrt(sum((np.abs(D[f"vA_{c}"]) ** 2).sum() for c in ("Ex", "Ey")) * dA_)
    ntB = math.sqrt(sum((np.abs(D[f"vB0_{c}"]) ** 2).sum() for c in ("Ex", "Ey")) * dA_)
    dens = {c: np.real(np.conj(D[f"vB0_{c}"]) * D[f"vA_{c}"]) * inB / (ntA * ntB) for c in ("Ex", "Ey", "Ez")}
    parts = {c: float(dens[c].sum() * dA_) for c in dens}
    tot_ = sum(parts.values())
    kparts = {c: V["kappa_overlap"] * parts[c] / tot_ for c in parts}
    R["kappa_by_component"] = kparts
    ez_share = float((np.abs(D["vA_Ez"]) ** 2).sum() / sum((np.abs(D[f"vA_{c}"]) ** 2).sum() for c in ("Ex", "Ey", "Ez")))
    R["vector"]["Ez_share_of_E2_A"] = ez_share
    p = Page(7, "Why is the scalar model 2x too strong for silicon?", (2, 2))
    ax = p.fig.add_axes([0.07, 0.24, 0.40, 0.50])
    labs = ["scalar\nmodel", "$E_x E_x$", "$E_y E_y$", "$E_z E_z$", "vector\ntotal"]
    vals = [S["kappa_overlap"], kparts["Ex"], kparts["Ey"], kparts["Ez"], V["kappa_overlap"]]
    ax.bar(range(5), vals, color=["#9fbbd9", "#4a7fb5", "#9ab8d6", B_COL, NAVY], width=0.65)
    for i, v_ in enumerate(vals):
        ax.text(i, v_ + (0.003 if v_ >= 0 else -0.008), f"{v_:+.4f}" if 0 < i < 4 else f"{v_:.4f}",
                ha="center", fontsize=11.5)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labs)
    ax.set_ylabel("contribution to $\\kappa_c$ (rad/µm)")
    ax.set_ylim(-0.055, 0.085)
    ax.set_title("the vector overlap, split by field component", loc="left")
    tag(ax, 3, 0.03, 1)
    vmax_ = max(np.abs(dens["Ex"]).max(), np.abs(dens["Ez"]).max())
    selx = (xs > xb - W / 2 - 0.02) & (xs < xb + W / 2 + 0.02)
    sely = (ys > -H / 2 - 0.02) & (ys < H / 2 + 0.02)
    maps_ax = []
    for k_, (c, ttl) in enumerate((("Ex", "$E_x E_x$ inside B"), ("Ez", "$E_z E_z$ inside B"))):
        axm = p.fig.add_axes([0.56, 0.53 - k_ * 0.30, 0.36, 0.20])
        im = axm.imshow(dens[c][np.ix_(sely, selx)], extent=[xs[selx][0], xs[selx][-1], ys[sely][0], ys[sely][-1]],
                        origin="lower", cmap="RdBu_r", vmin=-vmax_, vmax=vmax_, aspect="equal")
        axm.add_patch(Rectangle((xb - W / 2, -H / 2), W, H, fill=False, ec=B_COL, lw=1.5))
        axm.set_title(ttl, loc="left")
        axm.set_yticks([-0.1, 0, 0.1])
        if k_ == 0:
            axm.set_xticklabels([])
        maps_ax.append(axm)
    axm.set_xlabel("x (µm) across core B;  guide A is off to the left")
    cax = p.fig.add_axes([0.935, 0.25, 0.012, 0.47])
    cb = p.fig.colorbar(im, cax=cax)
    cb.set_ticks([-vmax_, 0, vmax_])
    cb.set_ticklabels(["−", "0", "+"])
    tag(axm, xb + W / 2 - 0.05, 0.0, 2)
    p.footer([f"A silicon mode's field lines tilt along z ($E_z$, {ez_share*100:.0f} % of $|E|^2$).  The scalar model has no $E_z$.",
              "Inside B, A's tilt and B's tilt point opposite ways: the $E_z E_z$ term is negative and cancels most of $E_x E_x$."],
             "same logic as steps 4 to 6, but the integrand is the full vector dot product.  From here on: full Maxwell.")
    paths.append(p.save("why_vector", pdf))

    # ================================================================ step 8
    p = Page(8, "What do $dA/dz = -j\\kappa_c B$ and $dB/dz = -j\\kappa_c A$ do over distance?")
    z = D["z"]
    Lc = V["L_c_exact"]
    PA, PB = np.abs(D["ode_A"]) ** 2, np.abs(D["ode_B"]) ** 2
    ax = p.fig.add_axes([0.06, 0.24, 0.56, 0.45])
    ax.plot(z, PA, color=A_COL, lw=2.8, label="$|A|^2$  coupled equations")
    ax.plot(z, PB, color=B_COL, lw=2.8, label="$|B|^2$  coupled equations")
    ax.plot(D["z_half"][::3], D["half_PA"][::3], "o", ms=5.5, mfc="white", mec=A_COL, label="exact field, A's half")
    ax.plot(D["z_half"][::3], D["half_PB"][::3], "o", ms=5.5, mfc="white", mec=B_COL, label="exact field, B's half")
    ax.plot(z, PA + PB, color=GREY, lw=1.2, ls="--")
    ax.text(Lc * 0.78, 1.025, "$|A|^2 + |B|^2$ = 1", ha="center", va="bottom", color=GREY, fontsize=11.5)
    ax.set_xticks([0, Lc / 2, Lc, 1.5 * Lc, 2 * Lc])
    ax.set_xticklabels(["0", "$L_c/2$", f"$L_c$ = {Lc:.0f} µm", "$3L_c/2$", "$2L_c$"])
    for m_ in (1, 2):
        ax.axvline(m_ * Lc, color=GREY, lw=0.9, ls=":")
    ax.set_xlim(0, z[-1])
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("z")
    ax.set_ylabel("fraction of launched power")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    tag(ax, Lc * 0.5, 0.2, 1)
    ax = p.fig.add_axes([0.70, 0.24, 0.27, 0.48])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#d5d9de", lw=1)
    ax.plot(np.real(D["ode_A"]), -np.imag(D["ode_B"]), color=P_COL, lw=2.4)
    for frac in (0, 0.5, 1.0, 1.5):
        i_ = int(np.argmin(np.abs(z - frac * Lc)))
        ax.plot(np.real(D["ode_A"][i_]), -np.imag(D["ode_B"][i_]), "o", color=P_COL, ms=7)
        ax.annotate("z = 0" if frac == 0 else f"{frac:g} $L_c$", (np.real(D["ode_A"][i_]), -np.imag(D["ode_B"][i_])),
                    xytext=(8, 6), textcoords="offset points", fontsize=11.5)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.5)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("A", color=A_COL, fontsize=14)
    ax.set_ylabel("jB", color=B_COL, fontsize=14)
    ax.set_title("A against jB", loc="left")
    tag(ax, -1.1, -1.1, 2)
    p.footer([f"Lines: the coupled equations solved from A = 1, B = 0.  Circles: the exact field.  All power is in B after "
              f"$L_c = \\pi/2\\kappa_c$ = {Lc:.1f} µm.",
              "The state runs round a circle at rate $\\kappa_c$: the oscillator $A'' = -\\kappa_c^2 A$.  "
              "The $-j$ means B starts 90° behind A."],
             "$P_A = \\cos^2\\kappa_c z$,  $P_B = \\sin^2\\kappa_c z$,  and $P_A + P_B = 1$: power moves, it is never lost.")
    paths.append(p.save("oscillation", pdf))

    # ================================================================ step 9
    p = Page(9, "Why does the power hop back and forth at all?", (1, 2))
    for key, title, nn, xpos, tg in (("P", "even supermode $e_+$", V["n_plus"], 0.03, 1),
                                     ("M", "odd supermode $e_-$", V["n_minus"], 0.52, 2)):
        fld = np.real(D[f"v{key}_Ex"])
        r3.height_surface(str(REND / f"s9_{key}.png"), X, Y, fld, coresAB, cmap="RdBu_r", zscale=0.34,
                          fmax=np.abs(fld).max(), core_style="outline")
        ax = p.fig.add_axes([xpos, 0.25, 0.45, 0.46])
        show_render(ax, REND / f"s9_{key}.png")
        ax.set_title(f"{title}     $n_{{eff}}$ = {nn:.4f}", loc="center", fontsize=15)
        tag(ax, 60, 40, tg)
    p.footer(["The true modes of the pair (field $E_x$; red +, blue −).  Neither lives in one guide.  "
              "The spikes at the sidewalls are the normal-E jump.",
              f"Light launched in A alone is {V['power_in_plus']*100:.1f} % $e_+$ + {V['power_in_minus']*100:.1f} % $e_-$:  "
              "they add in A and cancel in B."],
             f"the two supermodes travel at slightly different speeds ($n_+ - n_-$ = {V['n_plus']-V['n_minus']:.4f}).")
    paths.append(p.save("supermodes", pdf))

    p = Page(9, "Why does the power hop back and forth at all?", (2, 2))
    cpv, cmv = complex(D["sup_cp"]), complex(D["sup_cm"])
    bpv, bmv = float(D["beta_p"]), float(D["beta_m"])
    zt = np.linspace(0, 2 * Lc, 600)
    I = np.zeros((len(xs), len(zt)))
    for c_ in ("Ex", "Ey", "Ez"):
        f_ = (cpv * D[f"vP_{c_}"][iy0][:, None] * np.exp(-1j * bpv * zt)[None, :]
              + cmv * D[f"vM_{c_}"][iy0][:, None] * np.exp(-1j * bmv * zt)[None, :])
        I += np.abs(f_) ** 2
    ax = p.fig.add_axes([0.07, 0.22, 0.84, 0.52])
    im = ax.imshow(I / I.max(), extent=[0, 2 * Lc, xs[0], xs[-1]], origin="lower", aspect="auto",
                   cmap="inferno", vmin=0, vmax=1)
    for xc, col in ((xa, A_COL), (xb, B_COL)):
        for e_ in (-1, 1):
            ax.axhline(xc + e_ * W / 2, color=col, lw=1, ls="--")
    ax.set_yticks([xa, xb])
    ax.set_yticklabels(["A", "B"], fontsize=15, fontweight="bold")
    ax.set_ylim(-0.95, 0.95)
    for m_ in (1, 2):
        ax.axvline(m_ * Lc, color="white", lw=1, ls=":")
    ax.text(Lc, 0.82, f"$L_c = \\lambda_0 / 2(n_+ - n_-)$ = {Lc:.1f} µm", color="white", ha="center", fontsize=13)
    ax.set_xlabel("z (µm), along the coupler")
    ax.set_title("$|c_+ e_+ e^{-j\\beta_+ z} + c_- e_- e^{-j\\beta_- z}|^2$, seen from above (mid-height)", loc="left")
    p.fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("$|E|^2$ / max")
    tag(ax, Lc * 0.5, -0.8, 1)
    p.footer(["Where the two supermodes are in step the light is in A; half a slip later it is in B."],
             "'power hopping' is two supermodes beating, and $(\\beta_+ - \\beta_-)/2$ is the same $\\kappa_c$ as step 7.")
    paths.append(p.save("beat", pdf))

    # ================================================================ step 10
    if fd is None:
        p = Page(10, "Does a full 3-D Maxwell simulation agree?")
        p.fig.text(0.5, 0.5, "FDTD results not found.  Run fdtd3d.py, then run.py again.", ha="center", fontsize=18)
        p.footer(["(missing)"], "(missing)")
        paths.append(p.save("fdtd_missing", pdf))
    else:
        F, FJ = fd[0]
        zF, PAF, PBF = F["z"], F["PA"], F["PB"]
        P0 = (PAF + PBF)[zF >= 0].mean()
        R["fdtd"] = []
        for F_, J_ in fd:
            k_, mx_ = fit_kappa(F_, V["kappa_exact"])
            J_.update(kappa_fit=k_, L_c_fit=math.pi / (2 * k_), max_transfer=mx_)
            R["fdtd"].append(J_)
        kF = FJ["kappa_fit"]
        zsc = 1 / 18.0
        tz, tx, tI = F["top_z"], F["top_x"], F["top_I"]
        cuts = {float(z_): F["xsec_I"][i] for i, z_ in enumerate(F["xsec_z"])}
        cx, cy = F["xsec_x"], F["xsec_y"]
        kt = (tx > -1.1) & (tx < 1.1)
        kx, ky = (cx > -1.1) & (cx < 1.1), (cy > -0.5) & (cy < 0.5)
        anc = r3.coupler(str(REND / "s10_fdtd3d.png"), cx[kx], cy[ky],
                         lambda z_: cuts[z_][np.ix_(ky, kx)], list(cuts.keys()), float(zF.max()) + 0.5,
                         zscale_len=zsc, cores=coresAB, top_I=tI[kt], top_z=tz,
                         anchors={f"z{int(z_)}": (1.2, 0.35, z_ * zsc) for z_ in cuts}
                         | {"A0": (xa, 0.32, -0.15), "B0": (xb, 0.32, -0.15)})
        p = Page(10, "Does a full 3-D Maxwell simulation agree?", (1, 2))
        ax = p.fig.add_axes([0.04, 0.19, 0.92, 0.55])
        a = show_render(ax, REND / "s10_fdtd3d.png", anc)
        for z_ in cuts:
            ax.text(*a[f"z{int(z_)}"], f"z = {z_:.0f} µm", fontsize=12, ha="left", va="bottom")
        ax.text(*a["A0"], "A", color=A_COL, fontweight="bold", ha="center", fontsize=16)
        ax.text(*a["B0"], "B", color=B_COL, fontweight="bold", ha="center", fontsize=16)
        p.fig.text(0.035, 0.775, f"Meep 3-D FDTD:  |E|² on the mid-height plane and five cross-sections  "
                   f"(z drawn {1/zsc:.0f}x shorter)", fontsize=13.5, fontweight="bold")
        p.footer([f"Maxwell's equations stepped in time on a 3-D grid ({FJ['resolution_px_per_um']} px/µm).  "
                  "No coupled-mode theory goes in.",
                  "Light enters guide A.  Guide B starts at z = 0, so A = 1, B = 0 there.  Follow the bright core from slice to slice."],
                 "a brute-force Maxwell solve moves the light from A to B on its own.  Next page: by how much, and how fast?")
        paths.append(p.save("fdtd_3d", pdf))

        p = Page(10, "Does a full 3-D Maxwell simulation agree?", (2, 2))
        ax = p.fig.add_axes([0.07, 0.46, 0.86, 0.25])
        zz_ = np.linspace(0, zF.max(), 400)
        kc = V["kappa_exact"]
        ax.plot(zz_, np.cos(kc * zz_) ** 2, color=A_COL, lw=2.2, label="mode solver prediction, A")
        ax.plot(zz_, np.sin(kc * zz_) ** 2, color=B_COL, lw=2.2, label="mode solver prediction, B")
        kM = FJ["kappa_mpb_same_grid"]
        ax.plot(zz_, np.sin(kM * zz_) ** 2, color=B_COL, lw=1.4, ls="--",
                label=f"Meep's mode solver, same {FJ['resolution_px_per_um']} px/µm grid, B")
        ax.plot(zF, PAF / P0, "o", ms=5.5, mfc="white", mec=A_COL, label="FDTD, power in A's half")
        ax.plot(zF, PBF / P0, "o", ms=5.5, mfc="white", mec=B_COL, label="FDTD, power in B's half")
        ax.set_ylabel("fraction of power")
        ax.set_xlim(-1, zF.max() + 1)
        ax.set_ylim(-0.03, 1.05)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=10.5)
        tag(ax, 8, 0.5, 1)
        ax = p.fig.add_axes([0.07, 0.215, 0.86, 0.19])
        ax.imshow(tI[kt] / tI[kt].max(), extent=[tz[0], tz[-1], tx[kt][0], tx[kt][-1]], origin="lower",
                  aspect="auto", cmap="inferno", vmin=0, vmax=0.6)
        ax.set_xlim(-1, zF.max() + 1)
        ax.set_yticks([xa, xb])
        ax.set_yticklabels(["A", "B"], fontsize=13, fontweight="bold")
        ax.set_xlabel("z (µm)")
        tag(ax, 66, -0.7, 2)
        grid_txt = ",  ".join(f"{J_['resolution_px_per_um']} px/µm: {J_['L_c_fit']:.1f} µm"
                              for J_ in sorted(R["fdtd"], key=lambda j: j["resolution_px_per_um"]))
        p.footer([f"FDTD $L_c$ at {grid_txt}.   Converged mode solver: {V['L_c_exact']:.1f} µm.   The FDTD grid is not "
                  "converged yet.",
                  f"FDTD at {FJ['resolution_px_per_um']} px/µm (dots) tracks Meep's own mode solver on the same grid (dashed) to "
                  f"{kF/kM*100:.0f} %; $P_A + P_B$ is constant to {FJ['total_power_spread']*100:.2f} %."],
                 f"with no coupled-mode theory, Maxwell's equations move {FJ['max_transfer']*100:.1f} % of the light into B.  "
                 "The exact length is grid-limited here.")
        paths.append(p.save("fdtd_power", pdf))

    # ================================================================ step 11
    sw = R["gap_sweep"]
    gs = np.array([s_["gap_um"] for s_ in sw]) * 1e3
    kve = np.array([s_["vector"]["kappa_exact"] for s_ in sw])
    kvo = np.array([s_["vector"]["kappa_overlap"] for s_ in sw])
    p = Page(11, "Why does $\\kappa_c$ fall like $e^{-\\gamma g}$, and not $e^{-2\\gamma g}$?", (1, 2))
    ax = p.fig.add_axes([0.07, 0.22, 0.38, 0.52])
    ax.semilogy(gs, kve, "o-", color=NAVY, lw=2.2, ms=7, label="exact (supermodes)")
    ax.semilogy(gs, kvo, "s", color="#9fbbd9", ms=8, mfc="none", mew=2, label="overlap formula")
    ax.set_xlabel("gap g (nm)")
    ax.set_ylabel("$\\kappa_c$ (rad/µm)")
    ax.legend(frameon=False)
    ax.set_title("$\\kappa_c$ against gap (full Maxwell)", loc="left")
    sl = R["vector_kappa_slope_per_um"]
    tag(ax, 330, kve[1], 1)
    ax = p.fig.add_axes([0.57, 0.22, 0.38, 0.52])
    d_ = xl - (xa + W / 2)
    selr = (d_ > 0.01) & (d_ < 0.62)
    cpos, rate = local_rate(d_[selr], D["vA_Ex_line"][selr])
    ax.plot(cpos * 1e3, rate, color=A_COL, lw=2.6, label="field of A,  $|E|$")
    ax.plot(cpos * 1e3, 2 * rate, color=A_COL, lw=2, ls="--", label="power of A,  $|E|^2$")
    gm = (gs[1:] + gs[:-1]) / 2
    krate = -np.diff(np.log(kve)) / np.diff(gs * 1e-3)
    ax.plot(gm, krate, "o", color=NAVY, ms=9, label="coupling,  $\\kappa_c$")
    ax.set_xlabel("distance from A's sidewall  /  gap (nm)")
    ax.set_ylabel("decay rate (1/µm)")
    ax.set_ylim(0, 32)
    ax.set_xlim(0, 600)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("what $\\kappa_c$ decays like", loc="left")
    tag(ax, 60, 19, 2)
    p.footer([f"On a log axis $\\kappa_c$ is a straight line in gap: $\\kappa_c \\propto e^{{-\\gamma g}}$, γ = {sl:.1f} /µm.  "
              "The overlap formula tracks it at every gap.",
              "$\\kappa_c$ decays at the field's rate (solid), not the power's (dashed): the overlap integral contains $u_A$ once."],
             f"one extra decay length of gap (1/γ = {1e3/sl:.0f} nm) divides $\\kappa_c$ by e = 2.7.")
    paths.append(p.save("gap_decay", pdf))

    p = Page(11, "How much does a fabrication error in the gap matter?", (2, 2))
    gg = np.linspace(110, 190, 200)
    kint = np.exp(np.interp(gg, gs, np.log(kve)))
    k_at = lambda g_: float(np.exp(np.interp(g_, gs, np.log(kve))))  # noqa: E731
    L0 = math.pi / (2 * k_at(GAP * 1e3))
    ax = p.fig.add_axes([0.10, 0.22, 0.55, 0.52])
    ax.plot(gg, np.sin(kint * L0) ** 2 * 100, color=B_COL, lw=2.8, label=f"full crossover  (L = {L0:.1f} µm)")
    ax.plot(gg, np.sin(kint * L0 / 2) ** 2 * 100, color=P_COL, lw=2.8, label=f"50/50 splitter  (L = {L0/2:.1f} µm)")
    for dg in (-10, 10):
        v_ = math.sin(k_at(GAP * 1e3 + dg) * L0 / 2) ** 2 * 100
        ax.plot(GAP * 1e3 + dg, v_, "o", color=P_COL, ms=8)
        ax.annotate(f"{v_:.0f} %", (GAP * 1e3 + dg, v_), xytext=(10, -4 if dg > 0 else 4),
                    textcoords="offset points", fontsize=12.5, color=P_COL)
    ax.axvline(GAP * 1e3, color=GREY, lw=0.9, ls=":")
    ax.axvspan(GAP * 1e3 - 10, GAP * 1e3 + 10, color=GREY, alpha=0.08)
    ax.set_xlabel("fabricated gap (nm)   (design: 150 nm)")
    ax.set_ylabel("power reaching B (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    tag(ax, 185, 35, 1)
    km, kp = k_at(GAP * 1e3 - 10), k_at(GAP * 1e3 + 10)
    k0_ = k_at(GAP * 1e3)
    R["gap_error_10nm"] = {"kappa_minus10_pct": (km / k0_ - 1) * 100, "kappa_plus10_pct": (kp / k0_ - 1) * 100}
    p.footer([f"±10 nm of gap changes $\\kappa_c$ by {(km/k0_-1)*100:+.0f} % / {(kp/k0_-1)*100:+.0f} %.  The 50/50 splitter "
              f"becomes {math.sin(km*L0/2)**2*100:.0f}/{100-math.sin(km*L0/2)**2*100:.0f} or "
              f"{math.sin(kp*L0/2)**2*100:.0f}/{100-math.sin(kp*L0/2)**2*100:.0f}; the full crossover barely moves."],
             "coupler gaps are among the most tightly controlled dimensions on a chip.")
    paths.append(p.save("gap_error", pdf))

    pdf.close()
    return paths


if __name__ == "__main__":
    D = dict(np.load(OUT / "cache" / "solved.npz"))
    R = json.loads((OUT / "results.json").read_text())
    build_all(D, R, load_fdtd())
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
