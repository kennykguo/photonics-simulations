"""Animation: walk along the coupler and watch the light cross from A to B.

The field at each z is the exact superposition of the two solved supermodes,
c+ e+ exp(-j beta+ z) + c- e- exp(-j beta- z) (full Maxwell, femwell): for a
coupler that does not change along z this is the exact answer, not a model.
Writes out/coupler_walk.mp4 (and a small .gif).
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from cmt import H, W  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
A_COL, B_COL, INK, GREY = "#1f5fa8", "#c4501d", "#1d2733", "#6b7785"


def main(n_frames=240):
    D = dict(np.load(OUT / "cache" / "solved.npz"))
    R = json.loads((OUT / "results.json").read_text())
    V = R["vector"]
    xs, ys = D["xs"], D["ys"]
    xa, xb = float(D["xa"]), float(D["xb"])
    iy0 = int(np.argmin(np.abs(ys)))
    cp, cm = complex(D["sup_cp"]), complex(D["sup_cm"])
    bp, bm = float(D["beta_p"]), float(D["beta_m"])
    Lc = V["L_c_exact"]
    zmax = 2 * Lc
    comps = ("Ex", "Ey", "Ez")

    def xsec(z):
        ep, em = cp * np.exp(-1j * bp * z), cm * np.exp(-1j * bm * z)
        return sum(np.abs(ep * D[f"vP_{c}"] + em * D[f"vM_{c}"]) ** 2 for c in comps)

    zt = np.linspace(0, zmax, 700)
    top = sum(np.abs(cp * D[f"vP_{c}"][iy0][:, None] * np.exp(-1j * bp * zt)
                     + cm * D[f"vM_{c}"][iy0][:, None] * np.exp(-1j * bm * zt)) ** 2 for c in comps)
    vmax = max(xsec(0).max(), xsec(Lc).max())
    zh, PA, PB = D["z_half"], D["half_PA"], D["half_PB"]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.93, "Walking along the coupler", fontsize=24, fontweight="bold", color=INK)
    fig.text(0.04, 0.885, "full-Maxwell field of the 500 x 220 nm silicon pair, 150 nm gap, 1310 nm, light launched in A",
             fontsize=13, color=GREY)
    ax1 = fig.add_axes([0.04, 0.12, 0.42, 0.64])
    im = ax1.imshow(xsec(0) / vmax, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower", cmap="inferno",
                    vmin=0, vmax=1, aspect="equal")
    for xc, col in ((xa, A_COL), (xb, B_COL)):
        ax1.add_patch(Rectangle((xc - W / 2, -H / 2), W, H, fill=False, ec=col, lw=2))
    ax1.text(xa, -0.42, "A", color="white", fontsize=18, fontweight="bold", ha="center")
    ax1.text(xb, -0.42, "B", color="white", fontsize=18, fontweight="bold", ha="center")
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel("x (µm)", fontsize=12)
    ax1.set_ylabel("y (µm)", fontsize=12)
    ax1.set_title("cross-section |E|² at the current z", loc="left", fontsize=14, fontweight="bold")
    ztxt = ax1.text(-1.05, 0.40, "", color="white", fontsize=16, fontweight="bold", family="monospace")

    ax2 = fig.add_axes([0.52, 0.47, 0.45, 0.29])
    ax2.imshow(top / top.max(), extent=[0, zmax, xs[0], xs[-1]], origin="lower", aspect="auto", cmap="inferno")
    ax2.set_ylim(-0.95, 0.95)
    ax2.set_yticks([xa, xb])
    ax2.set_yticklabels(["A", "B"], fontsize=13, fontweight="bold")
    ax2.set_title("seen from above", loc="left", fontsize=14, fontweight="bold")
    l2 = ax2.axvline(0, color="white", lw=2)

    ax3 = fig.add_axes([0.52, 0.12, 0.45, 0.25])
    ax3.plot(zh, PA, color=A_COL, lw=2.6, label="power in A")
    ax3.plot(zh, PB, color=B_COL, lw=2.6, label="power in B")
    ax3.set_xlim(0, zmax)
    ax3.set_ylim(-0.03, 1.05)
    ax3.set_xlabel("z (µm)", fontsize=12)
    ax3.legend(frameon=False, loc="center right", fontsize=12)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    l3 = ax3.axvline(0, color=INK, lw=1.5)
    dA, = ax3.plot([0], [1], "o", color=A_COL, ms=9)
    dB, = ax3.plot([0], [0], "o", color=B_COL, ms=9)

    zf = np.linspace(0, zmax, n_frames)

    def upd(i):
        z = zf[i]
        im.set_data(xsec(z) / vmax)
        ztxt.set_text(f"z = {z:5.1f} µm")
        l2.set_xdata([z, z])
        l3.set_xdata([z, z])
        dA.set_data([z], [np.interp(z, zh, PA)])
        dB.set_data([z], [np.interp(z, zh, PB)])
        return im, ztxt, l2, l3, dA, dB

    anim = FuncAnimation(fig, upd, frames=n_frames, blit=False)
    anim.save(OUT / "coupler_walk.mp4", writer=FFMpegWriter(fps=24, bitrate=3000), dpi=100)
    anim.save(OUT / "coupler_walk.gif", writer=PillowWriter(fps=15), dpi=50)
    plt.close(fig)
    print("  wrote coupler_walk.mp4 / .gif")


if __name__ == "__main__":
    main()
