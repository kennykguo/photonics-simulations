"""Visualise n_eff and n_g for a silicon waveguide.

Run:   cd ~/photonics-simulations/visuals/01_neff_and_ng && ../../.venv/bin/python run.py
Out:   out/*.png, out/phase_vs_group.mp4, out/phase_vs_group.gif, out/results.json

Everything is computed here: the materials come from Sellmeier fits, the mode
comes from a full-vector finite-element solve, and the group index comes from
differentiating that solve with respect to wavelength. Nothing is typed in.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Rectangle

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import indices as ix  # noqa: E402

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- settings
LAM = 1.31        # um, the O-band carrier Lightmatter uses
WIDTH = 0.50      # um, strip width
HEIGHT = 0.220    # um, silicon device layer thickness
RING_L = 39.6     # um, round trip of the reference 6.3 um ring

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# one colour per concept, used in every figure
C_PHASE = "#2a78d6"   # blue  = n_eff, phase, crests
C_GROUP = "#eb6834"   # orange = n_g, group, envelope
C_INK = "#1a1a19"
C_MUTED = "#6b6a66"
C_CORE = "#cfe0f5"
C_ACCENT = "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d8d6d0",
    "axes.labelcolor": C_INK,
    "axes.titlecolor": C_INK,
    "axes.grid": True,
    "grid.color": "#e8e6e0",
    "grid.linewidth": 0.7,
    "text.color": C_INK,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "font.size": 10,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
})

R = {}   # everything printed ends up here and is written to out/results.json
t_start = time.time()


def say(msg: str) -> None:
    print(f"[{time.time() - t_start:6.1f}s] {msg}", flush=True)


# =========================================================================
# 1. Solve the waveguide
# =========================================================================
say("solving the 500 x 220 nm strip with femwell (full-vector FEM)")

n_si = float(ix.n_silicon(LAM))
n_ox = float(ix.n_silica(LAM))
mode = ix.strip_modes(LAM, WIDTH, HEIGHT, resolution=0.03, pad_um=1.0, num_modes=1)[0]
n_eff, n_g, slope = ix.group_index_strip(LAM, WIDTH, HEIGHT, dlam=0.02,
                                         resolution=0.03, pad_um=1.0)
v = ix.velocities(n_eff, n_g, LAM)

# the exact 1-D slab of the same thickness, for comparison
slab_neff = ix.slab_neff(LAM, HEIGHT, n_si, n_ox, 0)
slab_n0, slab_ng, _ = ix.group_index_slab(LAM, HEIGHT)

R["geometry"] = {"lambda_um": LAM, "width_um": WIDTH, "height_um": HEIGHT,
                 "n_silicon": n_si, "n_silica": n_ox}
R["strip_mode"] = {"n_eff": n_eff, "n_g": n_g, "dneff_dlambda_per_um": slope,
                   "te_fraction": mode.te_fraction,
                   "dneff_dnsi": mode.dneff_dnsi,
                   "energy_fraction_core": mode.energy_fraction_core,
                   "dlambda_dT_pm_per_K": LAM / n_g * mode.dneff_dnsi * 1.86e-4 * 1e6}
R["velocities"] = v
R["slab_reference"] = {"n_eff": slab_neff, "n_g": slab_ng}
R["bulk_group_index"] = {"silicon": ix.bulk_group_index(ix.n_silicon, LAM),
                         "silica": ix.bulk_group_index(ix.n_silica, LAM)}

say(f"  n_eff = {n_eff:.4f}   n_g = {n_g:.4f}   v_p/v_g = {n_g/n_eff:.3f}")
say(f"  v_phase = {v['v_phase_um_per_ps']:.1f} um/ps   "
    f"v_group = {v['v_group_um_per_ps']:.1f} um/ps")


# =========================================================================
# Figure 1: the mode, and what the two indices mean
# =========================================================================
say("figure 1: the mode and the two indices")

fig = plt.figure(figsize=(13.5, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.45)

# --- (a) the 2-D mode from femwell
ax = fig.add_subplot(gs[0, 0])
# the overall sign of an eigenmode is arbitrary and flips between solves, so
# plot the magnitude: it is the same picture every run and it is what "where the
# light is" actually means.
mode.mode_obj.plot_component("E", "x", part="abs", boundaries=True,
                             colorbar=False, ax=ax)
# femwell's default colormap is not diverging; a signed field needs one that is
# symmetric about zero so the sign of E_x is readable.
for coll in ax.collections:
    arr = coll.get_array()
    if arr is not None and len(np.asarray(arr)) > 1:
        a = np.abs(np.asarray(arr))
        coll.set_cmap("magma_r")
        coll.set_clim(0, float(a.max()))
        cb = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("|E$_x$| (a.u.)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
ax.add_patch(Rectangle((-WIDTH / 2, -HEIGHT / 2), WIDTH, HEIGHT, fill=False,
                       ec=C_INK, lw=1.4, zorder=6))
ax.set_xlim(-0.75, 0.75)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel("x across the width (um)")
ax.set_ylabel("y across the thickness (um)")
ax.set_title("(a) the actual mode: |E$_x$| of the quasi-TE\n"
             f"500 x 220 nm strip, {mode.te_fraction*100:.1f} % TE", fontsize=10)
ax.grid(False)

# --- (b) vertical cut: the exact slab solution of the same thickness
ax = fig.add_subplot(gs[0, 1])
xs = np.linspace(-0.5, 0.5, 1200)
prof = ix.slab_profile(LAM, HEIGHT, n_si, n_ox, slab_neff, xs)
ax.fill_between([-HEIGHT / 2, HEIGHT / 2], -0.05, 1.08, color=C_CORE,
                zorder=0, label="silicon core")
ax.plot(xs, prof, color=C_ACCENT, lw=2.2)
k0 = 2 * math.pi / LAM
gamma = k0 * math.sqrt(slab_neff**2 - n_ox**2)
decay_nm = 1000 / gamma
ax.annotate(f"tail decays in {decay_nm:.0f} nm",
            xy=(HEIGHT / 2 + 1 / gamma, math.cos(k0 * math.sqrt(n_si**2 - slab_neff**2)
                                                 * HEIGHT / 2) * math.exp(-1)),
            xytext=(0.21, 0.62), fontsize=9, color=C_MUTED,
            arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1))
ax.set_xlabel("y across the thickness (um)")
ax.set_ylabel("field F(y), normalised")
ax.set_title("(b) exact slab solution, same 220 nm\n"
             f"gives n$_{{eff}}$ = {slab_neff:.3f} with no side walls", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-0.05, 1.12)

# --- (c) the readout
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.set_title("(c) the two indices this mode has", fontsize=10)

rows = [
    ("n$_{eff}$ = " + u"β/k₀", f"{n_eff:.4f}", C_PHASE,
     "crests move at c/n$_{eff}$"),
    ("n$_g$ = n$_{eff}$ " + u"− λ dn$_{eff}$/dλ", f"{n_g:.4f}", C_GROUP,
     "the pulse moves at c/n$_g$"),
]
y = 0.88
for label, value, colour, meaning in rows:
    ax.add_patch(Rectangle((0.02, y - 0.13), 0.96, 0.17,
                           facecolor=colour, alpha=0.10, edgecolor=colour, lw=1.2,
                           transform=ax.transAxes, clip_on=False))
    ax.text(0.06, y, label, fontsize=11, color=colour, weight="bold",
            transform=ax.transAxes, va="center")
    ax.text(0.68, y, value, fontsize=15, color=colour, weight="bold",
            transform=ax.transAxes, va="center")
    ax.text(0.06, y - 0.085, meaning, fontsize=9, color=C_MUTED,
            transform=ax.transAxes, va="center")
    y -= 0.235

lines = [
    ("phase velocity  c/n$_{eff}$", f"{v['v_phase_um_per_ps']:.1f} um/ps", C_PHASE),
    ("group velocity  c/n$_g$", f"{v['v_group_um_per_ps']:.1f} um/ps", C_GROUP),
    ("crests outrun the pulse by", f"{n_g/n_eff:.2f} x", C_INK),
    ("guided wavelength  " + u"λ₀/n$_{eff}$", f"{v['guided_wavelength_nm']:.0f} nm", C_INK),
    (u"mode's grip on the silicon  \u2202n$_{eff}$/\u2202n$_{Si}$",
     f"{mode.dneff_dnsi:.2f}", C_INK),
    ("1 K of heating shifts " + u"\u03bb$_r$" + " by",
     f"{LAM / n_g * mode.dneff_dnsi * 1.86e-4 * 1e6:.0f} pm", C_INK),
]
y = 0.40
for label, value, colour in lines:
    ax.text(0.06, y, label, fontsize=9.5, color=C_MUTED, transform=ax.transAxes)
    ax.text(0.98, y, value, fontsize=9.5, color=colour, weight="bold",
            ha="right", transform=ax.transAxes)
    y -= 0.088

fig.suptitle("A silicon waveguide carries one mode with two different speeds",
             fontsize=13, y=1.03)
fig.savefig(OUT / "1_mode_and_indices.png")
plt.close(fig)


# =========================================================================
# Figure 2 and the animation: watch the crests outrun the pulse
# =========================================================================
say("figure 2 + animation: crests versus envelope")

C_UM_PS = ix.C0 * 1e-6           # speed of light, um/ps
omega = 2 * math.pi * C_UM_PS / LAM      # rad/ps
beta = 2 * math.pi * n_eff / LAM         # rad/um
v_p = omega / beta                       # um/ps
v_g = C_UM_PS / n_g                      # um/ps

Z0, SIGMA = 3.0, 1.2       # envelope start and width, um
Z_CREST0 = 2.0             # the crest we tag, one micron behind the peak
ZMAX, T_END = 14.0, 0.10   # um, ps
z = np.linspace(0, ZMAX, 3000)
k_crest = -beta * Z_CREST0 / (2 * math.pi)   # picks out that one crest


def field(t: float) -> np.ndarray:
    env = np.exp(-(((z - Z0 - v_g * t) / SIGMA) ** 2))
    return env * np.cos(omega * t - beta * z), env


def crest_z(t: float) -> float:
    return (omega * t - 2 * math.pi * k_crest) / beta


def env_z(t: float) -> float:
    return Z0 + v_g * t


R["animation"] = {
    "omega_rad_per_ps": omega, "beta_rad_per_um": beta,
    "v_phase_um_per_ps": v_p, "v_group_um_per_ps": v_g,
    "duration_ps": T_END,
    "crest_travel_um": crest_z(T_END) - crest_z(0),
    "envelope_travel_um": env_z(T_END) - env_z(0),
    "slip_um": (crest_z(T_END) - crest_z(0)) - (env_z(T_END) - env_z(0)),
}
R["animation"]["slip_in_guided_wavelengths"] = (
    R["animation"]["slip_um"] / (v["guided_wavelength_nm"] / 1000))


def draw_guide(ax):
    """A reminder that this is happening inside the guide.

    The vertical axis is field amplitude, not a transverse coordinate, so the
    band is only a frame: the two lines mark the walls of the silicon core seen
    edge on, and everything drawn between them is the field travelling along z.
    """
    ax.axhspan(-1.35, -1.0, color="#eef1f4", zorder=0)
    ax.axhspan(1.0, 1.35, color="#eef1f4", zorder=0)
    ax.axhline(-1.0, color="#9aa3ad", lw=1.2, zorder=1)
    ax.axhline(1.0, color="#9aa3ad", lw=1.2, zorder=1)
    ax.text(0.12, 1.17, "silica cladding", fontsize=7.5, color=C_MUTED, va="center")
    ax.text(0.12, -1.18, "silica cladding", fontsize=7.5, color=C_MUTED, va="center")
    ax.text(0.12, 0.86, "inside the silicon core", fontsize=7.5,
            color=C_MUTED, va="center")


# ---- static version: three snapshots plus the space-time diagram
fig = plt.figure(figsize=(12.5, 8.2))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5], hspace=0.78)

for i, t in enumerate([0.0, T_END / 2, T_END]):
    ax = fig.add_subplot(gs[i, 0])
    e, env = field(t)
    draw_guide(ax)
    ax.plot(z, env, color=C_GROUP, lw=1.4, ls="--", alpha=0.9)
    ax.plot(z, -env, color=C_GROUP, lw=1.4, ls="--", alpha=0.9)
    ax.plot(z, e, color=C_PHASE, lw=1.1)
    zc, ze = crest_z(t), env_z(t)
    ax.plot([zc], [np.exp(-(((zc - ze) / SIGMA) ** 2))], "o", color=C_PHASE,
            ms=9, mec="white", mew=1.5, zorder=5)
    ax.plot([ze], [1.0], "o", color=C_GROUP, ms=9, mec="white", mew=1.5, zorder=5)
    ax.axvline(Z_CREST0, color=C_PHASE, lw=0.8, ls=":", alpha=0.6)
    ax.axvline(Z0, color=C_GROUP, lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(0, ZMAX)
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([])
    ax.set_ylabel("E", rotation=0, labelpad=10)
    ax.set_title(f"t = {t*1000:.0f} fs      "
                 f"tagged crest at {zc:.2f} um, pulse peak at {ze:.2f} um      "
                 f"crest is {zc-ze:+.2f} um relative to the peak",
                 fontsize=10, loc="left")
    if i == 0:
        ax.legend(handles=[
            plt.Line2D([], [], color=C_PHASE, lw=1.6,
                       label="field E(z,t): one tagged crest (dot) moves at c/n$_{eff}$"),
            plt.Line2D([], [], color=C_GROUP, lw=1.6, ls="--",
                       label="envelope: its peak (dot) moves at c/n$_g$"),
        ], loc="upper right", fontsize=8.5, ncol=1)
    if i == 2:
        ax.set_xlabel("z along the waveguide (um)")

ax = fig.add_subplot(gs[3, 0])
ts = np.linspace(0, T_END, 200)
ax.plot(ts * 1000, [crest_z(t) for t in ts], color=C_PHASE, lw=2.4,
        label=f"a wave crest: slope = c/n$_{{eff}}$ = {v_p:.0f} um/ps")
ax.plot(ts * 1000, [env_z(t) for t in ts], color=C_GROUP, lw=2.4,
        label=f"the pulse envelope: slope = c/n$_g$ = {v_g:.0f} um/ps")
ax.fill_between(ts * 1000, [env_z(t) for t in ts], [crest_z(t) for t in ts],
                color=C_PHASE, alpha=0.08)
ax.annotate(
    f"after {T_END*1000:.0f} fs the crest has gained "
    f"{R['animation']['slip_um']:.2f} um\n"
    f"= {R['animation']['slip_in_guided_wavelengths']:.1f} guided wavelengths",
    xy=(T_END * 1000, (crest_z(T_END) + env_z(T_END)) / 2),
    xytext=(T_END * 1000 * 0.46, 3.1), fontsize=9.5, color=C_INK,
    arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.1))
ax.set_xlabel("time (fs)")
ax.set_ylabel("distance along the guide (um)")
ax.set_title("Same thing as a distance-time plot: two different slopes, "
             "one mode", fontsize=10, loc="left", pad=10)
ax.legend(loc="upper left", fontsize=9)

fig.suptitle(
    f"n$_{{eff}}$ = {n_eff:.3f} and n$_g$ = {n_g:.3f} are two different speeds in "
    f"the same waveguide: the crests run {n_g/n_eff:.2f} x faster than the pulse",
    fontsize=13, y=0.965)
fig.savefig(OUT / "2_phase_vs_group.png")
plt.close(fig)

# ---- the animation
fig, (axf, axt) = plt.subplots(2, 1, figsize=(11, 6.2),
                               gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.42})

draw_guide(axf)
(line_e,) = axf.plot([], [], color=C_PHASE, lw=1.1)
(line_env_p,) = axf.plot([], [], color=C_GROUP, lw=1.5, ls="--")
(line_env_m,) = axf.plot([], [], color=C_GROUP, lw=1.5, ls="--")
(dot_c,) = axf.plot([], [], "o", color=C_PHASE, ms=11, mec="white", mew=1.6, zorder=5)
(dot_e,) = axf.plot([], [], "o", color=C_GROUP, ms=11, mec="white", mew=1.6, zorder=5)
axf.axvline(Z_CREST0, color=C_PHASE, lw=0.8, ls=":", alpha=0.5)
axf.axvline(Z0, color=C_GROUP, lw=0.8, ls=":", alpha=0.5)
axf.set_xlim(0, ZMAX)
axf.set_ylim(-1.35, 1.35)
axf.set_yticks([])
axf.set_xlabel("z along the waveguide (um)")
axf.set_ylabel("E", rotation=0, labelpad=10)
axf.legend(handles=[
    plt.Line2D([], [], color=C_PHASE, lw=1.8, marker="o", ms=7,
               label=f"tagged crest, c/n$_{{eff}}$ = {v_p:.0f} um/ps"),
    plt.Line2D([], [], color=C_GROUP, lw=1.8, ls="--", marker="o", ms=7,
               label=f"pulse envelope, c/n$_g$ = {v_g:.0f} um/ps"),
], loc="upper right", fontsize=9)
title = axf.set_title("", fontsize=11, loc="left")

(tr_c,) = axt.plot([], [], color=C_PHASE, lw=2.4)
(tr_e,) = axt.plot([], [], color=C_GROUP, lw=2.4)
axt.set_xlim(0, T_END * 1000)
axt.set_ylim(0, ZMAX)
axt.set_xlabel("time (fs)")
axt.set_ylabel("distance (um)")
axt.set_title("distance travelled: the steeper line is the crest", fontsize=10, loc="left")
gap_txt = axt.text(0.98, 0.06, "", transform=axt.transAxes, ha="right",
                   fontsize=10, color=C_INK, weight="bold")

N_FRAMES = 200
frame_t = np.linspace(0, T_END, N_FRAMES)


def animate(i):
    t = frame_t[i]
    e, env = field(t)
    line_e.set_data(z, e)
    line_env_p.set_data(z, env)
    line_env_m.set_data(z, -env)
    zc, ze = crest_z(t), env_z(t)
    dot_c.set_data([zc], [np.exp(-(((zc - ze) / SIGMA) ** 2))])
    dot_e.set_data([ze], [1.0])
    tt = frame_t[: i + 1]
    tr_c.set_data(tt * 1000, [crest_z(s) for s in tt])
    tr_e.set_data(tt * 1000, [env_z(s) for s in tt])
    title.set_text(
        f"t = {t*1000:5.1f} fs      crest {zc:5.2f} um      pulse {ze:5.2f} um      "
        f"crest is {zc-ze:+.2f} um from the peak")
    gap_txt.set_text(f"crest has gained {zc - Z_CREST0 - (ze - Z0):+.2f} um")
    return (line_e, line_env_p, line_env_m, dot_c, dot_e, tr_c, tr_e, title, gap_txt)


anim = FuncAnimation(fig, animate, frames=N_FRAMES, interval=40, blit=False)

import imageio_ffmpeg
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
anim.save(OUT / "phase_vs_group.mp4", writer=FFMpegWriter(fps=25, bitrate=2400))
say("  wrote phase_vs_group.mp4")
anim.save(OUT / "phase_vs_group.gif", writer=PillowWriter(fps=20), dpi=62)
say("  wrote phase_vs_group.gif")
plt.close(fig)


# =========================================================================
# Figure 3: where n_g comes from, graphically
# =========================================================================
say("figure 3: n_g as the intercept of the tangent to n_eff(lambda)")

lams = np.arange(1.20, 1.46, 0.02)
neffs = np.array([ix.neff_strip(l, WIDTH, HEIGHT, resolution=0.04, pad_um=0.8)
                  for l in lams])
R["wavelength_sweep"] = {"lambda_um": lams.tolist(), "n_eff": neffs.tolist()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

ax1.plot(lams, neffs, "o-", color=C_PHASE, ms=4,
         label="n$_{eff}$(" + u"λ" + ") from the FEM solver")
tangent_l = np.linspace(0, 1.48, 50)
ax1.plot(tangent_l, n_eff + slope * (tangent_l - LAM), color=C_GROUP, lw=1.8, ls="--",
         label="tangent at 1310 nm")
ax1.plot([0], [n_g], "o", color=C_GROUP, ms=11, mec="white", mew=1.5, zorder=5)
ax1.plot([LAM], [n_eff], "o", color=C_PHASE, ms=11, mec="white", mew=1.5, zorder=5)
ax1.annotate(f"intercept = n$_g$ = {n_g:.3f}", xy=(0, n_g), xytext=(0.18, 4.35),
             fontsize=10, color=C_GROUP, weight="bold",
             arrowprops=dict(arrowstyle="->", color=C_GROUP, lw=1.2))
ax1.annotate(f"n$_{{eff}}$ = {n_eff:.3f}\nat 1310 nm", xy=(LAM, n_eff),
             xytext=(0.82, 2.15), fontsize=10, color=C_PHASE, weight="bold",
             arrowprops=dict(arrowstyle="->", color=C_PHASE, lw=1.2))
ax1.set_xlim(0, 1.5)
ax1.set_ylim(1.9, 4.6)
ax1.set_xlabel(u"wavelength λ (um)")
ax1.set_ylabel("index")
ax1.set_title("n$_g$ = n$_{eff}$ " + u"− λ" + " dn$_{eff}$/d" + u"λ"
              + " is the tangent's intercept at " + u"λ" + " = 0",
              fontsize=10.5, loc="left")
ax1.legend(loc="lower left", fontsize=9)

ngs = []
for l in lams:
    _, g, _ = ix.group_index_strip(l, WIDTH, HEIGHT, dlam=0.02,
                                   resolution=0.04, pad_um=0.8)
    ngs.append(g)
ngs = np.array(ngs)
R["wavelength_sweep"]["n_g"] = ngs.tolist()

ax2.plot(lams * 1000, neffs, "o-", color=C_PHASE, ms=4, label="n$_{eff}$ (phase)")
ax2.plot(lams * 1000, ngs, "o-", color=C_GROUP, ms=4, label="n$_g$ (group)")
ax2.axvline(1310, color=C_MUTED, lw=1, ls=":")
ax2.fill_between(lams * 1000, neffs, ngs, color=C_GROUP, alpha=0.08)
ax2.annotate("the gap is " + u"−λ" + " dn$_{eff}$/d" + u"λ"
             + f" = {-LAM*slope:.2f}",
             xy=(1310, (n_eff + n_g) / 2), xytext=(1218, 3.72), fontsize=9.5,
             color=C_INK, arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1))
ax2.set_xlabel("wavelength (nm)")
ax2.set_ylabel("index")
ax2.set_title("Both indices across the O band: n$_g$ is always the larger one",
              fontsize=10.5, loc="left")
ax2.legend(loc="center right", fontsize=9)

fig.suptitle("Where the group index comes from: the slope of n$_{eff}$ against wavelength",
             fontsize=13, y=1.02)
fig.savefig(OUT / "3_where_ng_comes_from.png")
plt.close(fig)


# =========================================================================
# Figure 4: both indices as the waveguide changes shape
# =========================================================================
say("figure 4: sweeping the waveguide width")

widths = np.arange(0.30, 0.76, 0.05)
w_neff, w_ng = [], []
for w in widths:
    a, b, _ = ix.group_index_strip(LAM, float(w), HEIGHT, dlam=0.02,
                                   resolution=0.04, pad_um=0.8)
    w_neff.append(a)
    w_ng.append(b)
w_neff, w_ng = np.array(w_neff), np.array(w_ng)
R["width_sweep"] = {"width_um": widths.tolist(), "n_eff": w_neff.tolist(),
                    "n_g": w_ng.tolist()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

ax1.plot(widths * 1000, w_neff, "o-", color=C_PHASE, ms=5, label="n$_{eff}$ (phase)")
ax1.plot(widths * 1000, w_ng, "o-", color=C_GROUP, ms=5, label="n$_g$ (group)")
ax1.axhline(slab_neff, color=C_PHASE, ls=":", lw=1.2)
ax1.text(720, slab_neff + 0.03, f"infinitely wide slab: {slab_neff:.3f}",
         fontsize=8, color=C_PHASE, ha="right")
ax1.axvline(WIDTH * 1000, color=C_MUTED, lw=1, ls=":")
ax1.text(WIDTH * 1000 + 10, w_ng.max() - 0.12, "the 500 nm\nreference guide",
         fontsize=8.5, color=C_MUTED, va="top")
ax1.set_xlabel("waveguide width (nm)")
ax1.set_ylabel("index")
ax1.set_title("Wider guide: n$_{eff}$ rises, n$_g$ falls. They are not the same "
              "quantity.", fontsize=10.5, loc="left")
ax1.legend(loc="center right", fontsize=9)

ax2.plot(widths * 1000, ix.C0 * 1e-6 / w_neff, "o-", color=C_PHASE, ms=5,
         label="phase velocity c/n$_{eff}$")
ax2.plot(widths * 1000, ix.C0 * 1e-6 / w_ng, "o-", color=C_GROUP, ms=5,
         label="group velocity c/n$_g$")
ax2.axvline(WIDTH * 1000, color=C_MUTED, lw=1, ls=":")
ax2.set_xlabel("waveguide width (nm)")
ax2.set_ylabel("velocity (um/ps)")
ax2.set_title("The same sweep as speeds: the geometry sets both, differently",
              fontsize=10.5, loc="left")
ax2.legend(loc="center right", fontsize=9)

fig.suptitle("Both indices are properties of the waveguide's shape, not just its material",
             fontsize=13, y=1.02)
fig.savefig(OUT / "4_width_sweep.png")
plt.close(fig)


# =========================================================================
# Figure 5: which index does what in the capstone ring
# =========================================================================
say("figure 5: what each index controls in a microring")

m_order = n_eff * RING_L / LAM
m_int = round(m_order)
lam_res = n_eff * RING_L / m_int                     # a resonance near 1310 nm
fsr_correct = LAM**2 / (n_g * RING_L)                # the right answer
fsr_wrong = LAM**2 / (n_eff * RING_L)                # what n_eff would give
R["ring"] = {
    "round_trip_um": RING_L,
    "mode_order_m": m_order, "m_rounded": m_int,
    "resonance_near_1310_nm": lam_res * 1000,
    "fsr_nm_using_ng": fsr_correct * 1000,
    "fsr_nm_if_you_wrongly_used_neff": fsr_wrong * 1000,
    "error_percent": (fsr_wrong - fsr_correct) / fsr_correct * 100,
    "round_trip_time_ps": n_g * RING_L / (ix.C0 * 1e-6),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [0.85, 1.5], "wspace": 0.25})

# ---- (a) the ring, drawn tangent to its bus
ax1.axis("off")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title("(a) one ring, two jobs", fontsize=11, loc="left")
cx, cy, rr = 0.50, 0.60, 0.19
ax1.add_patch(plt.Circle((cx, cy), rr, fill=False, lw=4, color="#9aa3ad"))
ax1.plot([0.06, 0.94], [cy - rr - 0.035, cy - rr - 0.035], lw=4, color="#9aa3ad")
ax1.text(cx, cy, f"L = {RING_L} um", ha="center", va="center", fontsize=9.5,
         color=C_MUTED)
ax1.text(0.06, cy - rr - 0.10, "bus waveguide", fontsize=8.5, color=C_MUTED)
ax1.text(0.02, 0.985,
         "n$_{eff}$ decides WHERE a resonance sits",
         fontsize=10, color=C_PHASE, va="top", weight="bold")
ax1.text(0.02, 0.925,
         f"m = n$_{{eff}}$L/" + u"\u03bb" + f" = {m_order:.2f}" + u"  \u2192  " + f"m = {m_int}\n"
         f"puts one resonance at {lam_res*1000:.1f} nm",
         fontsize=9, color=C_PHASE, va="top")
ax1.text(0.02, 0.26,
         "n$_g$ decides HOW FAR APART they sit",
         fontsize=10, color=C_GROUP, va="top", weight="bold")
ax1.text(0.02, 0.20,
         u"FSR = \u03bb\u00b2/(n$_g$L) = " + f"{fsr_correct*1000:.2f} nm\n"
         f"one lap takes n$_g$L/c = {R['ring']['round_trip_time_ps']:.3f} ps",
         fontsize=9, color=C_GROUP, va="top")

# ---- (b) the resonance comb, right and wrong
lam_axis = np.linspace(1.298, 1.332, 6000)
ch0, ch_sp, n_ch = 1.3092, 0.00112, 8          # 8 channels on the 200 GHz grid
ch_span = (n_ch - 1) * ch_sp


def comb(fsr, width_um=0.000374):
    """Through-port transmission of a ring with the given free spectral range."""
    t = np.ones_like(lam_axis)
    for k in range(-6, 7):
        c = lam_res + k * fsr
        t *= 1 - 0.97 / (1 + ((lam_axis - c) / (width_um / 2)) ** 2)
    return t


ax2.axvspan((ch0 - ch_sp / 2) * 1000, (ch0 + ch_span + ch_sp / 2) * 1000,
            color=C_ACCENT, alpha=0.10, zorder=0)
for k in range(n_ch):
    ax2.axvline((ch0 + k * ch_sp) * 1000, color=C_ACCENT, lw=0.9, alpha=0.75, zorder=1)
ax2.plot(lam_axis * 1000, comb(fsr_correct), color=C_GROUP, lw=1.9, zorder=3)
for k in range(-3, 4):
    c = (lam_res + k * fsr_wrong) * 1000
    if lam_axis[0] * 1000 <= c <= lam_axis[-1] * 1000:
        ax2.axvline(c, color=C_PHASE, lw=1.7, ls="--", alpha=0.9, zorder=2)

ax2.annotate("", xy=(lam_res * 1000, 1.13),
             xytext=((lam_res + fsr_correct) * 1000, 1.13),
             arrowprops=dict(arrowstyle="<->", color=C_GROUP, lw=1.6))
ax2.text((lam_res + fsr_correct / 2) * 1000, 1.165,
         f"real FSR {fsr_correct*1000:.2f} nm", ha="center", fontsize=9,
         color=C_GROUP, weight="bold")
ax2.text((ch0 + ch_span / 2) * 1000, 0.09,
         f"the 8 channels span {ch_span*1000:.2f} nm", ha="center", fontsize=8.5,
         color="#0f7a55")

ax2.plot([], [], color=C_GROUP, lw=1.9,
         label=f"resonances where they really are, FSR = " + u"\u03bb\u00b2" +
               f"/(n$_g$L) = {fsr_correct*1000:.2f} nm")
ax2.plot([], [], color=C_PHASE, lw=1.7, ls="--",
         label=f"where you would put them using n$_{{eff}}$: {fsr_wrong*1000:.2f} nm "
               f"({R['ring']['error_percent']:.0f} % too far apart)")
ax2.plot([], [], color=C_ACCENT, lw=0.9, label="the 8 laser channels, 200 GHz apart")
ax2.set_xlabel("wavelength (nm)")
ax2.set_ylabel("through-port transmission")
ax2.set_ylim(-0.03, 1.30)
ax2.set_xlim(lam_axis[0] * 1000, lam_axis[-1] * 1000)
ax2.legend(loc="center left", fontsize=8.5, ncol=1, borderaxespad=0.8)
ax2.set_title("(b) the FSR has to clear the channel band, so it must come from n$_g$",
              fontsize=11, loc="left")

fig.suptitle("Why the capstone needs both numbers: n$_{eff}$ places the resonances, "
             "n$_g$ spaces them", fontsize=13, y=1.02)
fig.savefig(OUT / "5_ring_capstone.png")
plt.close(fig)


# =========================================================================
say("writing results.json")
R["runtime_s"] = time.time() - t_start
(OUT / "results.json").write_text(json.dumps(R, indent=2))

print("\n" + "=" * 68)
print(f"  n_eff = {n_eff:.4f}    crests move at {v['v_phase_um_per_ps']:.1f} um/ps")
print(f"  n_g   = {n_g:.4f}    the pulse moves at {v['v_group_um_per_ps']:.1f} um/ps")
print(f"  ratio = {n_g/n_eff:.3f}      guided wavelength {v['guided_wavelength_nm']:.0f} nm")
print(f"  ring: m = {m_order:.2f}, FSR = {fsr_correct*1000:.2f} nm "
      f"(n_eff would say {fsr_wrong*1000:.2f} nm, {R['ring']['error_percent']:.0f} % off)")
print("=" * 68)
print(f"  {len(list(OUT.glob('*')))} files in {OUT}")
print(f"  done in {R['runtime_s']:.0f} s")
