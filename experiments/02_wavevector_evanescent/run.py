"""02_wavevector_evanescent: the wavevector, its components, and what happens when one
component is forced to be imaginary.

Notes sections 4 (wavevector, transverse/longitudinal, k_x^2 + beta^2 = n^2 k0^2) and
17 (k_x = -j*gamma turns e^{-j k_x x} into e^{-gamma x}). Everything is a numpy
evaluation of ONE formula, E = Re{ exp(j(wt - k_x x - k_z z)) }, with k_x real or
imaginary. Nothing is solved; the point is to *see* the same formula change character.

Run from this directory:
    cd experiments/02_wavevector_evanescent && ../../.venv/bin/python run.py
Outputs go to out/ (figures, two mp4 videos with PNG contact sheets, results.json, tools.json).
"""
import sys, pathlib, json, time, math, subprocess, shutil
import matplotlib
matplotlib.use("Agg")                     # headless: before pyplot is ever imported
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE, C0, MU0
from common.params import k0_per_um, omega_rad_s
OUT = pathlib.Path(__file__).resolve().parent / "out"
if OUT.exists():                           # regenerate everything from scratch: no stale files survive
    shutil.rmtree(OUT)
OUT.mkdir()
use_style()

FFMPEG = "/opt/homebrew/bin/ffmpeg"
matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG
import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyArrowPatch, Rectangle

T_START = time.time()
RESULTS = {}

# ----------------------------------------------------------------------------- constants
lam0_nm = REF.lambda_nm                   # 1310 nm vacuum wavelength (to try 1550 nm, replace REF.lambda_nm here)
lam0_um = lam0_nm * 1e-3
k0 = k0_per_um(lam0_nm)                   # rad/um
omega = omega_rad_s(lam0_nm)              # rad/s
T_fs = 2 * math.pi / omega * 1e15         # optical period in fs
n1, n2 = REF.n_si, REF.n_sio2             # silicon core, silica cladding
neff_ref = REF.neff                       # 2.5 textbook reference
neff_tag = f"{neff_ref:.1f}".replace(".", "p")   # "2p5": used in the results.json key names below
neff_prop = min(1.0, 0.7 * n2)            # the propagating (real k_x) example in Part D: 1.0 for silica, 0.7*n2 if n2 is lowered
k_sio2 = n2 * k0                          # |k| for a plane wave in silica (rad/um)
lam_sio2_um = 2 * math.pi / k_sio2        # = lam0 / n2

RESULTS["lambda0_nm"] = lam0_nm
RESULTS["k0_rad_per_um"] = k0
RESULTS["period_fs"] = T_fs
RESULTS["n_sio2"] = n2
RESULTS["k_sio2_rad_per_um"] = k_sio2
RESULTS["lambda_in_silica_nm"] = lam_sio2_um * 1e3
RESULTS["lambda_in_silica_expected_nm"] = lam0_nm / n2

print(f"lambda0 = {lam0_nm:.0f} nm, k0 = {k0:.4f} rad/um, T = {T_fs:.3f} fs")
print(f"silica: |k| = n2 k0 = {k_sio2:.4f} rad/um, lambda = {lam_sio2_um*1e3:.1f} nm")


# ----------------------------------------------------------------------------- the one formula
def kx_of(kz, n=n2):
    """Transverse wavenumber from the dispersion relation k_x^2 = n^2 k0^2 - k_z^2 (notes 4, 17).

    Returns a real positive k_x (wave travelling toward +x) when k_z < n k0, and the
    purely imaginary branch k_x = -j*gamma when k_z > n k0, so that e^{-j k_x x} = e^{-gamma x}
    decays for x > 0 (the choice that stays finite far from the interface, notes 17).
    """
    kx2 = (n * k0) ** 2 - kz ** 2
    if kx2 >= 0:
        return math.sqrt(kx2)
    return -1j * math.sqrt(-kx2)


def field(X, Z, wt, kx, kz):
    """E(x,z,t) = Re{ exp(j(wt - k_x x - k_z z)) }, the e^{j w t} convention (notes 3, 4).
    kx may be complex; numpy does the rest."""
    return np.real(np.exp(1j * (wt - kx * X - kz * Z)))


# ============================================================================= PART A
# A tilted plane wave in silica: k at angle theta from the z axis.
theta_deg = 30.0
theta = math.radians(theta_deg)
kz_A = k_sio2 * math.cos(theta)
kx_A = k_sio2 * math.sin(theta)
lam_z_um = 2 * math.pi / kz_A            # wavelength measured along z (> lambda)
lam_x_um = 2 * math.pi / kx_A            # wavelength measured along x (> lambda)
v_p = C0 / n2                            # phase velocity along k
v_p_along_z = omega / (kz_A * 1e6)       # crest speed along the z axis (m/s)
v_p_along_x = omega / (kx_A * 1e6)

RESULTS["partA_theta_deg"] = theta_deg
RESULTS["partA_kz_rad_per_um"] = kz_A
RESULTS["partA_kx_rad_per_um"] = kx_A
RESULTS["partA_k_magnitude_check"] = math.hypot(kx_A, kz_A) / k0      # should equal n2
RESULTS["partA_lambda_z_nm"] = lam_z_um * 1e3
RESULTS["partA_lambda_z_expected_nm"] = lam_sio2_um * 1e3 / math.cos(theta)
RESULTS["partA_lambda_x_nm"] = lam_x_um * 1e3
RESULTS["partA_lambda_x_expected_nm"] = lam_sio2_um * 1e3 / math.sin(theta)
RESULTS["partA_neff_of_tilted_wave"] = kz_A / k0                       # n2 cos(theta) < n2
RESULTS["partA_vp_m_per_s"] = v_p
RESULTS["partA_vp_along_z_m_per_s"] = v_p_along_z
RESULTS["partA_vp_along_z_over_c"] = v_p_along_z / C0

# grid: z horizontal, x vertical
nz, nx = 420, 420
z = np.linspace(0, 2.5, nz)
x = np.linspace(-1.25, 1.25, nx)
Z, X = np.meshgrid(z, x)
E_A0 = field(X, Z, 0.0, kx_A, kz_A)

# Measure the wavelengths along each axis from the computed field (zero-crossing spacing).
def measured_period(line, coord):
    s = np.sign(line)
    idx = np.where(np.diff(s) != 0)[0]
    zc = coord[idx] - line[idx] * (coord[idx + 1] - coord[idx]) / (line[idx + 1] - line[idx])
    return 2 * np.mean(np.diff(zc))     # two zero crossings per period

lam_z_meas = measured_period(E_A0[np.argmin(abs(x)), :], z)
lam_x_meas = measured_period(E_A0[:, np.argmin(abs(z))], x)
RESULTS["partA_lambda_z_measured_nm"] = lam_z_meas * 1e3
RESULTS["partA_lambda_x_measured_nm"] = lam_x_meas * 1e3
# wavelength along k: sample the field along the k direction
s_par = np.linspace(0, 2.0, 2000)
E_par = field(s_par * math.sin(theta), s_par * math.cos(theta), 0.0, kx_A, kz_A)
lam_par_meas = measured_period(E_par, s_par)
RESULTS["partA_lambda_along_k_measured_nm"] = lam_par_meas * 1e3
RESULTS["partA_lambda_z_agreement_pct"] = 100 * lam_z_meas / lam_z_um
RESULTS["partA_lambda_x_agreement_pct"] = 100 * lam_x_meas / lam_x_um
RESULTS["partA_lambda_along_k_agreement_pct"] = 100 * lam_par_meas / lam_sio2_um
RESULTS["partA_lambda_z_over_lambda"] = lam_z_um / lam_sio2_um          # = 1/cos(theta) > 1
print(f"Part A: theta={theta_deg} deg; kz={kz_A:.4f}, kx={kx_A:.4f} rad/um; "
      f"lambda_z={lam_z_um*1e3:.1f} nm (measured {lam_z_meas*1e3:.1f}), "
      f"lambda_x={lam_x_um*1e3:.1f} nm (measured {lam_x_meas*1e3:.1f}), "
      f"lambda along k measured {lam_par_meas*1e3:.1f} nm")


BOX = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)


def draw_plane_wave_annotations(ax):
    """k arrow with its components, one phase front, and the three wavelengths."""
    o = np.array([0.6, -0.6])
    scale = 0.12                                             # um per (rad/um), just for drawing
    kvec = np.array([kz_A, kx_A]) * scale
    ax.add_patch(FancyArrowPatch(o, o + kvec, arrowstyle="-|>", mutation_scale=18,
                                 color=PALETTE["ink"], lw=2.5, zorder=6))
    ax.add_patch(FancyArrowPatch(o, o + [kvec[0], 0], arrowstyle="-|>", mutation_scale=14,
                                 color=SERIES[3], lw=2, ls="--", zorder=6))
    ax.add_patch(FancyArrowPatch(o + [kvec[0], 0], o + kvec, arrowstyle="-|>", mutation_scale=14,
                                 color=SERIES[2], lw=2, ls="--", zorder=6))
    ax.text(*(o + kvec * 0.55 + [-0.08, 0.12]), "k", fontsize=13, weight="bold", color=PALETTE["ink"])
    ax.text(*(o + [kvec[0] * 0.5, -0.14]), f"k_z = β = {kz_A:.2f} rad/µm", color=SERIES[3], fontsize=9, ha="center", bbox=BOX)
    ax.text(*(o + [kvec[0] + 0.05, kvec[1] * 0.5]), f"k_x = {kx_A:.2f} rad/µm", color=SERIES[2], fontsize=9, bbox=BOX)
    # a phase front through the tip of k: perpendicular to k
    tip = o + kvec
    n_hat = np.array([-math.sin(theta), math.cos(theta)])   # (z, x) components perpendicular to k
    p0, p1 = tip - 0.6 * n_hat, tip + 0.6 * n_hat
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=PALETTE["ink"], lw=1.2, ls=":", zorder=6)
    ax.text(*(p1 + 0.04 * n_hat), "constant-phase front, perpendicular to k", fontsize=8, color=PALETTE["ink2"], bbox=BOX, ha="left", va="bottom")
    # wavelengths along z (at top) and along x (at right)
    x_top = 1.05
    ax.annotate("", xy=(1.2 + lam_z_um, x_top), xytext=(1.2, x_top),
                arrowprops=dict(arrowstyle="<->", color=SERIES[3], lw=1.5))
    ax.text(1.2 + lam_z_um / 2, x_top + 0.05, f"2π/k_z = {lam_z_um*1e3:.0f} nm", color=SERIES[3], ha="center", fontsize=9, bbox=BOX)
    z_right = 2.3
    ax.annotate("", xy=(z_right, -1.1 + lam_x_um), xytext=(z_right, -1.1),
                arrowprops=dict(arrowstyle="<->", color=SERIES[2], lw=1.5))
    ax.text(z_right - 0.04, -1.1 + lam_x_um / 2, f"2π/k_x = {lam_x_um*1e3:.0f} nm", color=SERIES[2],
            rotation=90, va="center", ha="right", fontsize=9, bbox=BOX)
    # wavelength along k: drawn on the far side of the front (beyond its upper end), clear of the front label,
    # the k_z / k_x labels and the crest paths of the video
    k_dir = np.array([math.cos(theta), math.sin(theta)])
    a = tip + n_hat * 0.9 - k_dir * 0.55
    b = a + k_dir * lam_sio2_um
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<->", color=PALETTE["ink"], lw=1.5))
    ax.text(*((a + b) / 2 + n_hat * 0.09), f"λ = 2π/|k| = {lam_sio2_um*1e3:.0f} nm", rotation=theta_deg,
            ha="center", va="center", fontsize=9, color=PALETTE["ink"], bbox=BOX)


# ---- static figure A
fig, ax = plt.subplots(figsize=(9, 6.6))
im = ax.imshow(E_A0, extent=[z[0], z[-1], x[0], x[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
draw_plane_wave_annotations(ax)
ax.set_xlabel("z (µm)"); ax.set_ylabel("x (µm)")
ax.set_title(f"A plane wave in silica (n = {n2}) with k tilted {theta_deg:.0f}° from z: the wavelength read along z or x is longer than λ", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("E(x, z, t=0) / E₀")
ax.grid(False)
fig.tight_layout(); fig.savefig(OUT / "plane_wave_tilted.png"); plt.close(fig)

# ---- animation A: the pattern moves along k; a crest tracked along k and along z
n_frames_A = 90
periods_A = 2.3                     # long enough to see the crest cross most of the frame, short enough to stay in view
# The circle starts on the phase-0 crest line, offset u0 along the front (perpendicular to k) so its
# path stays clear of the annotations; the square is where the SAME crest line cuts the z axis.
u0 = -0.5
n_front = np.array([-math.sin(theta), math.cos(theta)])   # (z, x) unit vector along a phase front


def crest_positions(wt):
    s = wt / k_sio2                          # distance the crest has moved along k: wt - |k| s = 0
    circle = np.array([s * math.cos(theta), s * math.sin(theta)]) + u0 * n_front
    square = np.array([wt / kz_A, 0.0])      # same crest line, on the z axis: wt - k_z z = 0
    return circle, square



fig, ax = plt.subplots(figsize=(9, 6.6))
im = ax.imshow(E_A0, extent=[z[0], z[-1], x[0], x[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
draw_plane_wave_annotations(ax)
ax.set_xlabel("z (µm)"); ax.set_ylabel("x (µm)"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("E / E₀")
# crest markers: start on the crest through the origin (phase 0) at t=0
crest_k, = ax.plot([], [], "o", color=SERIES[5], ms=9, mec=PALETTE["ink"], zorder=7, label="crest, followed along k (v = c/n)")
crest_z, = ax.plot([], [], "s", color=SERIES[1], ms=9, mec=PALETTE["ink"], zorder=7, label="same crest, where it cuts the z axis (v = ω/k_z > c/n)")
ax.legend(loc="lower left", fontsize=8, facecolor=PALETTE["surface"], framealpha=0.9, frameon=True)
title = ax.set_title("", fontsize=10)


def anim_A(i):
    t_fs = periods_A * T_fs * i / n_frames_A
    wt = omega * t_fs * 1e-15
    im.set_data(field(X, Z, wt, kx_A, kz_A))
    circle, square = crest_positions(wt)
    crest_k.set_data([circle[0]], [circle[1]])
    crest_z.set_data([square[0]], [square[1]])
    title.set_text(f"E = E₀ cos(ωt − k_x x − k_z z), t = {t_fs:5.2f} fs = {t_fs/T_fs:4.2f} T   (T = {T_fs:.2f} fs, λ₀ = {lam0_nm:.0f} nm)")
    return im, crest_k, crest_z, title


ani = FuncAnimation(fig, anim_A, frames=n_frames_A, blit=False)
ani.save(OUT / "plane_wave_tilted.mp4", writer=FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
plt.close(fig)

# contact sheet A
sel = [4, 21, 38, 55, 72]      # start a few frames in so the square (z = ωt/k_z) sits clear of the panel's left edge
fig, axes = plt.subplots(1, len(sel), figsize=(15, 3.6))
for ax_, i in zip(axes, sel):
    t_fs = periods_A * T_fs * i / n_frames_A
    wt = omega * t_fs * 1e-15
    ax_.imshow(field(X, Z, wt, kx_A, kz_A), extent=[z[0], z[-1], x[0], x[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    circle, square = crest_positions(wt)
    ax_.plot([circle[0]], [circle[1]], "o", color=SERIES[5], mec=PALETTE["ink"], ms=8, clip_on=False, label="crest, followed along k (v = c/n₂)")
    ax_.plot([square[0]], [square[1]], "s", color=SERIES[1], mec=PALETTE["ink"], ms=8, clip_on=False, label="same crest on the z axis (v = ω/k_z)")
    ax_.set_xlim(z[0], z[-1]); ax_.set_ylim(x[0], x[-1])     # markers must not autoscale the panel
    ax_.set_title(f"t = {t_fs:.2f} fs = {t_fs/T_fs:.1f} T", fontsize=9); ax_.grid(False)
    ax_.set_xlabel("z (µm)")
axes[0].set_ylabel("x (µm)")
axes[0].legend(loc="lower left", fontsize=6.5, facecolor=PALETTE["surface"], framealpha=0.9, frameon=True)
fig.suptitle("Frames of plane_wave_tilted.mp4: the crest (circle) moves along k at c/n; its trace on the z axis (square) runs faster", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "plane_wave_tilted_frames.png"); plt.close(fig)
print("Part A figures and video written")


# ============================================================================= PART B
# Sweep k_z = n_eff k0 past n2 k0 in the silica half-space x > 0. Interface at x = 0.
# Physically, k_z is imposed by the wave on the silicon side (n_eff = n1 sin(theta_i), notes 24).
z_B = np.linspace(0, 1.6, 320)
x_B = np.linspace(0, 1.0, 320)
Z_B_MAX = float(z_B[-1])
ZB, XB = np.meshgrid(z_B, x_B)


def gamma_of(neff, n_clad=n2):
    return k0 * math.sqrt(neff ** 2 - n_clad ** 2)


gamma_ref = gamma_of(neff_ref)                 # rad/um at n_eff = 2.5 in silica
decay_ref_nm = 1e3 / gamma_ref
RESULTS[f"gamma_per_um_neff{neff_tag}_silica"] = gamma_ref
RESULTS[f"decay_length_nm_neff{neff_tag}_silica"] = decay_ref_nm
RESULTS["decay_length_expected_nm"] = 102.0    # value quoted in the brief / textbook (for n_eff = 2.5, silica)
print(f"Part B: n_eff={neff_ref} in silica: gamma = {gamma_ref:.4f} 1/um, 1/gamma = {decay_ref_nm:.1f} nm")

# Numerical check that the complex-k_x branch of the formula IS e^{-gamma x} e^{-j beta z}
kz_ref = neff_ref * k0
kx_ref = kx_of(kz_ref)
RESULTS[f"kx_at_neff{neff_tag}_rad_per_um"] = [kx_ref.real, kx_ref.imag]
RESULTS["dispersion_check_kx2_plus_kz2_over_k02"] = float(np.real(kx_ref ** 2 + kz_ref ** 2) / k0 ** 2)   # should be n2^2
RESULTS["dispersion_check_expected_n2_squared"] = n2 ** 2
wt_chk = 1.234
E_branch = field(XB, ZB, wt_chk, kx_ref, kz_ref)
E_explicit = np.exp(-gamma_ref * XB) * np.cos(wt_chk - kz_ref * ZB)
RESULTS["identity_check_max_abs_diff"] = float(np.max(np.abs(E_branch - E_explicit)))
print(f"  k_x = {kx_ref}, kx^2+kz^2 = {RESULTS['dispersion_check_kx2_plus_kz2_over_k02']:.4f} k0^2 (n2^2 = {n2**2:.4f})")
print(f"  max |Re e^(j(wt-kx x-kz z)) - e^(-gamma x)cos(wt-kz z)| = {RESULTS['identity_check_max_abs_diff']:.2e}")

# Fit the decay length from the computed field envelope (max over one z period, per x)
env = np.max(np.abs(E_branch), axis=1)

def expo(xx, A, L):
    return A * np.exp(-xx / L)

popt, _ = curve_fit(expo, x_B, env, p0=[1.0, 0.1])
RESULTS["decay_length_fitted_from_field_nm"] = popt[1] * 1e3
RESULTS["decay_length_fit_agreement_pct"] = 100 * popt[1] * 1e3 / decay_ref_nm
print(f"  exponential fit of the field envelope: 1/gamma = {popt[1]*1e3:.2f} nm ({RESULTS['decay_length_fit_agreement_pct']:.2f} % of analytic)")

# The two angles that matter in silicon (notes 24): critical angle and the angle that gives n_eff = 2.5
theta_c_deg = math.degrees(math.asin(n2 / n1))
theta_i_ref_deg = math.degrees(math.asin(neff_ref / n1))
RESULTS["critical_angle_silicon_silica_deg"] = theta_c_deg
RESULTS[f"incidence_angle_in_silicon_for_neff{neff_tag}_deg"] = theta_i_ref_deg
print(f"  critical angle Si/SiO2 = {theta_c_deg:.2f} deg; n_eff {neff_ref} <-> incidence {theta_i_ref_deg:.2f} deg inside silicon")

# decay lengths at the other n_eff values that appear in the notes and in experiment 08
for label, nn in [("neff2p7_solver", 2.7), ("neff2p97_notes22", 2.97)]:
    RESULTS[f"decay_length_nm_{label}_silica"] = 1e3 / gamma_of(nn)
RESULTS["decay_length_nm_neff2p97_notes23_expected"] = 80.0      # notes 23 quote 80 nm for n_eff = 2.97
RESULTS["decay_length_neff2p97_agreement_with_notes23_pct"] = 100 * RESULTS["decay_length_nm_neff2p97_notes22_silica"] / 80.0
RESULTS[f"decay_length_nm_neff{neff_tag}_air"] = 1e3 / gamma_of(neff_ref, 1.0)
print(f"  1/gamma: n_eff 2.7 -> {RESULTS['decay_length_nm_neff2p7_solver_silica']:.1f} nm; "
      f"n_eff 2.97 -> {RESULTS['decay_length_nm_neff2p97_notes22_silica']:.1f} nm (notes 23 says 80 nm: {RESULTS['decay_length_neff2p97_agreement_with_notes23_pct']:.1f} %); "
      f"air cladding at {neff_ref} -> {RESULTS[f'decay_length_nm_neff{neff_tag}_air']:.1f} nm")

# ---- animation B: sweep n_eff from 0.3 to neff_ref while time runs, then hold at neff_ref
neff_path = np.concatenate([np.linspace(0.3, neff_ref, 200), np.full(60, neff_ref)])
n_frames_B = len(neff_path)
fig = plt.figure(figsize=(12.4, 6.6))   # 1612 x 858 px at 130 dpi: both even, as libx264 yuv420p requires
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.5, wspace=0.3, left=0.06, right=0.98, top=0.9, bottom=0.09)
axF = fig.add_subplot(gs[:, 0]); axK = fig.add_subplot(gs[0, 1]); axP = fig.add_subplot(gs[1, 1])

# The image covers exactly the computed rows x_B = 0..1 um; the silicon block is drawn BELOW it (x < 0) after
# widening the y limits, so that the row at the drawn interface really is x = 0 (the same extent as the contact sheet).
imB = axF.imshow(np.zeros_like(XB), extent=[z_B[0], z_B[-1], x_B[0], x_B[-1]], origin="lower",
                 cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
axF.add_patch(Rectangle((0, -0.25), Z_B_MAX, 0.25, facecolor="#c9c4b8", edgecolor="none", zorder=3))
axF.text(Z_B_MAX / 2, -0.13, "silicon (n₁ = 3.50): sets the shared k_z = β along the interface", ha="center", va="center", fontsize=8.5, zorder=4)
axF.axhline(0, color=PALETTE["ink"], lw=1.2, zorder=4)
axF.text(0.02, 0.93, f"silica (n₂ = {n2})", fontsize=9, color=PALETTE["ink2"], bbox=BOX, zorder=5)
axF.set_xlabel("z (µm), along the interface"); axF.set_ylabel("x (µm), away from the interface"); axF.grid(False)
axF.set_ylim(-0.25, 1.0)
karrow = FancyArrowPatch((0.3, 0.3), (0.3, 0.3), arrowstyle="-|>", mutation_scale=18, color=PALETTE["ink"], lw=2.5, zorder=6)
axF.add_patch(karrow)
titleF = axF.set_title("", fontsize=9.5, loc="left")
fig.colorbar(imB, ax=axF, fraction=0.03, pad=0.02, label="E / E₀")
fig.suptitle("One formula, E = Re{e^{j(ωt − k_x x − k_z z)}}, evaluated in the silica while k_z = n_eff·k₀ is swept through n₂k₀", fontsize=10.5)

# right top: k_x^2 vs n_eff  (the dispersion relation, a parabola through zero at n2)
neff_axis_max = max(2.7, neff_ref + 0.2)     # keeps the swept dot inside the panel if neff_ref is raised (experiment #2)
neff_axis = np.linspace(0, neff_axis_max, 400)
kx2_axis = (n2 * k0) ** 2 - (neff_axis * k0) ** 2
axK.plot(neff_axis, kx2_axis, color=SERIES[0], label="k_x² = n₂²k₀² − k_z²")
axK.axhline(0, color=PALETTE["ink2"], lw=1)
axK.axvline(n2, color=SERIES[4], ls="--", lw=1.2, label=f"k_z = n₂k₀ (n_eff = {n2})")
axK.axvspan(0, n2, color=SERIES[0], alpha=0.06); axK.axvspan(n2, neff_axis_max, color=SERIES[1], alpha=0.08)
axK.text(0.7, -45, "k_x real:\nphase rotates in x", fontsize=8, ha="center", color=SERIES[0])
axK.text(2.05, 22, "k_x = −jγ:\namplitude decays in x", fontsize=8, ha="center", color=SERIES[1])
axK.set_ylim(-125, 60)
ptK, = axK.plot([], [], "o", color=PALETTE["ink"], ms=8, zorder=6)
axK.set_xlabel("n_eff = k_z / k₀"); axK.set_ylabel("k_x² (rad²/µm²)")
axK.set_title("Dispersion relation in the cladding: k_x² changes sign at n_eff = n₂", fontsize=9)
axK.legend(fontsize=7, loc="lower left")

# right bottom: profile along x at the z of the current crest
lineP, = axP.plot([], [], color=SERIES[0], lw=2, label="E(x) at the crest z")
envP, = axP.plot([], [], color=SERIES[1], lw=1.5, ls="--", label="envelope e^{−γx}")
axP.axhline(0, color=PALETTE["ink2"], lw=0.8)
axP.set_xlim(0, 1.0); axP.set_ylim(-1.1, 1.1)
axP.set_xlabel("x (µm) into the silica"); axP.set_ylabel("E / E₀")
txtP = axP.text(0.98, 0.9, "", transform=axP.transAxes, ha="right", va="top", fontsize=9)
axP.legend(fontsize=7, loc="lower right")
axP.set_title("Field across x: cosine when k_x is real, exponential when it is imaginary", fontsize=9)


def anim_B(i):
    neff = neff_path[i]
    kz = neff * k0
    kx = kx_of(kz)
    t_fs = 0.02 * T_fs * i               # slow time so the fronts visibly drift
    wt = omega * t_fs * 1e-15
    E = field(XB, ZB, wt, kx, kz)
    imB.set_data(E)
    kx2 = (n2 * k0) ** 2 - kz ** 2
    ptK.set_data([neff], [kx2])
    # profile at the z where cos(wt - kz z) = 1 nearest z = 1 um
    z_crest = (wt - 2 * math.pi * round((wt - kz * 1.0) / (2 * math.pi))) / kz if kz > 0 else 0.0
    z_crest = min(max(z_crest, 0), Z_B_MAX)
    j = int(np.argmin(np.abs(z_B - z_crest)))
    lineP.set_data(x_B, E[:, j])
    if kx2 >= 0:
        ang = math.degrees(math.atan2(kx.real, kz))           # angle of k from the z axis
        envP.set_data([], [])
        lam_x = 2 * math.pi / kx.real if kx.real > 1e-9 else float("inf")
        txtP.set_text(f"k_x = {kx.real:.2f} rad/µm real\n2π/k_x = {lam_x*1e3:.0f} nm\nk tilted {ang:.0f}° from z")
        L = 0.35
        karrow.set_positions((0.3, 0.3), (0.3 + L * math.cos(math.radians(ang)), 0.3 + L * math.sin(math.radians(ang))))
        karrow.set_visible(True)
        titleF.set_text(f"n_eff = k_z/k₀ = {neff:.2f} < n₂ = {n2}: a propagating plane wave;\nthe fronts tilt toward grazing as k_z grows   (t = {t_fs:.2f} fs)")
    else:
        g = -kx.imag
        envP.set_data(x_B, np.exp(-g * x_B))
        txtP.set_text(f"k_x = −j{g:.2f} rad/µm imaginary\nγ = {g:.2f} /µm,  1/γ = {1e3/g:.0f} nm\nno phase change in x, only decay")
        karrow.set_visible(False)
        titleF.set_text(f"n_eff = {neff:.2f} > n₂ = {n2}: the SAME formula gives e^(−γx)·cos(ωt − βz):\ndecay in x, still travelling in z   (t = {t_fs:.2f} fs)")
    return imB, ptK, lineP, envP, txtP, titleF, karrow


ani = FuncAnimation(fig, anim_B, frames=n_frames_B, blit=False)
ani.save(OUT / "evanescent_sweep.mp4", writer=FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
plt.close(fig)

# contact sheet B
neff_mid = 1.8 if n2 < 1.8 < neff_ref else round(0.5 * (n2 + neff_ref), 2)
sel_neff = [0.5, 1.0, round(n2 - 0.05, 2), n2, neff_mid, neff_ref]   # 0.5, 1.0, 1.40, 1.45, 1.8, 2.5 by default
fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
for ax_, nn in zip(axes.ravel(), sel_neff):
    kz = nn * k0; kx = kx_of(kz)
    E = field(XB, ZB, 0.0, kx, kz)
    ax_.imshow(E, extent=[z_B[0], z_B[-1], 0, x_B[-1]], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax_.axhline(0, color=PALETTE["ink"], lw=1.5)
    ax_.grid(False); ax_.set_xlabel("z (µm)"); ax_.set_ylabel("x (µm)")
    if isinstance(kx, complex):
        g = -kx.imag
        ax_.set_title(f"n_eff = {nn:.2f} > n₂: k_x = −j{g:.2f}/µm, 1/γ = {1e3/g:.0f} nm", fontsize=9)
    else:
        ang = math.degrees(math.atan2(kx, kz))
        ax_.set_title(f"n_eff = {nn:.2f}: k_x = {kx:.2f} rad/µm, k at {ang:.0f}° from z", fontsize=9)
fig.suptitle("Frames of evanescent_sweep.mp4 (silica half-space, interface at x = 0): increasing k_z tilts the fronts, then kills the transverse oscillation", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "evanescent_sweep_frames.png"); plt.close(fig)
print("Part B video written")

# ---- static figure: decay length vs n_eff for silica and air cladding
neff_s = np.linspace(1.46, 3.5, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(neff_s, 1e3 / (k0 * np.sqrt(neff_s ** 2 - n2 ** 2)), color=SERIES[0], label=f"silica cladding, n₂ = {n2}")
ax.plot(neff_s, 1e3 / (k0 * np.sqrt(neff_s ** 2 - 1.0 ** 2)), color=SERIES[1], label="air cladding, n₂ = 1.0")
for nn, lab, c, off in [(2.5, "textbook n_eff 2.5 → %.0f nm", SERIES[3], (-30, 40)),
                        (2.7, "strip-solver n_eff ≈ 2.7 → %.0f nm", SERIES[2], (10, 22)),
                        (2.97, "notes §22 n_eff 2.97 → %.0f nm", SERIES[4], (10, -28))]:
    d = 1e3 / gamma_of(nn)
    ax.plot([nn], [d], "o", color=c, ms=8, zorder=5)
    ax.annotate(lab % d, (nn, d), xytext=off, textcoords="offset points", fontsize=8, color=c,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
ax.axvline(n2, color=PALETTE["muted"], ls=":", lw=1)
ax.text(n2 - 0.025, 20, "n_eff = n₂: 1/γ → ∞ (grazing)", fontsize=8, color=PALETTE["muted"], rotation=90, va="bottom", ha="center")   # left of the dotted line, clear of the curve
ax.set_ylim(0, 700); ax.set_xlim(n2 - 0.05, 3.5)
ax.set_xlabel("n_eff = β / k₀"); ax.set_ylabel("decay length 1/γ (nm)")
ax.set_title("The evanescent tail gets shorter the further β sits above n₂k₀: 1/γ = λ₀ / (2π √(n_eff² − n₂²))", fontsize=10)
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "decay_length_vs_neff.png"); plt.close(fig)

# ---- decay of the tail on a log scale with the 200 nm gap marked
xx = np.linspace(0, 0.5, 200)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.semilogy(xx * 1e3, np.exp(-gamma_ref * xx), color=SERIES[0], label="field e^{−γx}")
ax.semilogy(xx * 1e3, np.exp(-2 * gamma_ref * xx), color=SERIES[1], label="intensity e^{−2γx}")
for X0, lab in [(decay_ref_nm, "1/γ"), (200, "200 nm gap")]:
    ax.axvline(X0, color=PALETTE["muted"], ls=":", lw=1)
    ax.text(X0 + 3, 2e-4, lab, fontsize=8, color=PALETTE["ink2"], rotation=90, va="bottom")
f200 = math.exp(-gamma_ref * 0.2)
ax.plot([decay_ref_nm], [math.exp(-1)], "o", color=SERIES[0]); ax.annotate(f"e⁻¹ = 0.368 at {decay_ref_nm:.0f} nm", (decay_ref_nm, math.exp(-1)), xytext=(10, 5), textcoords="offset points", fontsize=8)
ax.plot([200], [f200], "o", color=SERIES[0]); ax.annotate(f"field {f200:.3f} at the bus", (200, f200), xytext=(10, 5), textcoords="offset points", fontsize=8)
ax.plot([200], [f200 ** 2], "o", color=SERIES[1]); ax.annotate(f"intensity {f200**2:.4f}", (200, f200 ** 2), xytext=(10, -12), textcoords="offset points", fontsize=8)
ax.set_xlabel("x, distance into the silica (nm)"); ax.set_ylabel("relative to the value at the interface")
ax.set_title(f"At n_eff = {neff_ref} the tail loses a factor e every {decay_ref_nm:.0f} nm;\nat the 200 nm ring-bus gap only {100*f200:.0f} % of the field ({100*f200**2:.0f} % of the intensity) is left", fontsize=10)
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "tail_log_scale.png"); plt.close(fig)
RESULTS["field_fraction_at_200nm_gap"] = f200
RESULTS["intensity_fraction_at_200nm_gap"] = f200 ** 2


# ============================================================================= PART C: capstone
# Coupling scales with the field of the ring mode where the bus sits: kappa ∝ e^{-gamma gap}.
gap_ref_nm = 200.0
gaps = np.linspace(100, 400, 301)
kappa2_ref = REF.kappa2
kappa_rel = np.exp(-gamma_ref * (gaps - gap_ref_nm) * 1e-3)
kappa2 = kappa2_ref * kappa_rel ** 2
d_gap = 10.0
kappa_plus = math.exp(-gamma_ref * (-d_gap) * 1e-3) - 1      # gap 10 nm too small -> kappa up
kappa_minus = math.exp(-gamma_ref * (+d_gap) * 1e-3) - 1     # gap 10 nm too large -> kappa down
RESULTS["capstone_gap_ref_nm"] = gap_ref_nm
RESULTS["capstone_kappa2_ref"] = kappa2_ref
RESULTS["capstone_gamma_times_10nm"] = gamma_ref * d_gap * 1e-3
RESULTS["capstone_kappa_change_pct_gap_minus10nm"] = 100 * kappa_plus
RESULTS["capstone_kappa_change_pct_gap_plus10nm"] = 100 * kappa_minus
RESULTS["capstone_kappa_change_pct_linearised"] = 100 * gamma_ref * d_gap * 1e-3
RESULTS["capstone_kappa2_at_gap_190nm"] = kappa2_ref * (1 + kappa_plus) ** 2
RESULTS["capstone_kappa2_at_gap_210nm"] = kappa2_ref * (1 + kappa_minus) ** 2
RESULTS["capstone_kappa2_change_pct_linearised"] = 100 * 2 * gamma_ref * d_gap * 1e-3
RESULTS["capstone_gap_nm_for_kappa_to_double"] = math.log(2) / gamma_ref * 1e3
RESULTS["capstone_t_coupler_at_190nm"] = math.sqrt(1 - RESULTS["capstone_kappa2_at_gap_190nm"])
RESULTS["capstone_t_coupler_at_210nm"] = math.sqrt(1 - RESULTS["capstone_kappa2_at_gap_210nm"])
RESULTS["capstone_a_round_trip_ref"] = REF.a_round_trip
# How much does thermal drift change gamma itself?  gamma^2 = k0^2 (n_eff^2 - n2^2), so
# d(gamma)/gamma = (n_eff dn_eff - n2 dn2) / (n_eff^2 - n2^2); dn_eff/dT ~ Gamma dn_Si/dT + (1 - Gamma) dn_SiO2/dT.
dneff_dT = REF.confinement * REF.dn_si_dT + (1 - REF.confinement) * REF.dn_sio2_dT
dgamma_rel_per_K = (neff_ref * dneff_dT - n2 * REF.dn_sio2_dT) / (neff_ref ** 2 - n2 ** 2)
dT_range = REF.ambient_max_c - REF.ambient_min_c
RESULTS["capstone_dneff_dT_per_K"] = dneff_dT
RESULTS["capstone_dgamma_over_gamma_pct_per_K"] = 100 * dgamma_rel_per_K
RESULTS["capstone_dgamma_over_gamma_pct_over_ambient_range"] = 100 * dgamma_rel_per_K * dT_range
RESULTS["capstone_resonance_shift_nm_over_ambient_range"] = REF.dlambda_dT_pm_per_K * dT_range * 1e-3
RESULTS["capstone_kappa_change_brief_pct"] = 10.0                 # the brief's round number "10 % in kappa"
RESULTS["capstone_kappa_change_agreement_pct"] = 100 * RESULTS["capstone_kappa_change_pct_linearised"] / 10.0
RESULTS["decay_length_agreement_with_brief_pct"] = 100 * decay_ref_nm / RESULTS["decay_length_expected_nm"]
print(f"Part C: gamma*10nm = {RESULTS['capstone_gamma_times_10nm']:.4f}; kappa {100*kappa_plus:+.1f} % (gap -10 nm), {100*kappa_minus:+.1f} % (gap +10 nm); "
      f"kappa^2 {kappa2_ref:.3f} -> {RESULTS['capstone_kappa2_at_gap_190nm']:.3f} / {RESULTS['capstone_kappa2_at_gap_210nm']:.3f}; "
      f"t {RESULTS['capstone_t_coupler_at_190nm']:.3f} / {RESULTS['capstone_t_coupler_at_210nm']:.3f} vs a = {REF.a_round_trip}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.semilogy(gaps, kappa_rel, color=SERIES[0], label="field coupling κ / κ(200 nm) ∝ e^{−γ·gap}")
ax1.semilogy(gaps, kappa_rel ** 2, color=SERIES[1], label="power coupling κ² / κ²(200 nm) ∝ e^{−2γ·gap}")
ax1.axvspan(gap_ref_nm - d_gap, gap_ref_nm + d_gap, color=SERIES[4], alpha=0.15, label="±10 nm fabrication error")
ax1.axvline(gap_ref_nm, color=PALETTE["muted"], ls=":", lw=1)
ax1.set_xlabel("ring–bus gap (nm)"); ax1.set_ylabel("relative to the 200 nm design")
ax1.set_title(f"With 1/γ = {decay_ref_nm:.0f} nm, every 10 nm of gap is ≈{100*gamma_ref*d_gap*1e-3:.0f} % of κ (≈{200*gamma_ref*d_gap*1e-3:.0f} % of κ²)", fontsize=10)
ax1.legend(fontsize=8)
g_zoom = np.linspace(180, 220, 81)
k2_zoom = kappa2_ref * np.exp(-2 * gamma_ref * (g_zoom - gap_ref_nm) * 1e-3)
ax2.plot(g_zoom, k2_zoom, color=SERIES[1], label="κ² (power coupling)")
ax2.axhline(1 - REF.a_round_trip ** 2, color=SERIES[3], ls="--", label=f"critical coupling κ² = 1 − a² = {1-REF.a_round_trip**2:.3f} (a = {REF.a_round_trip})")
ax2.axvspan(gap_ref_nm - d_gap, gap_ref_nm + d_gap, color=SERIES[4], alpha=0.15)
for gg in (190, 200, 210):
    kk = kappa2_ref * math.exp(-2 * gamma_ref * (gg - gap_ref_nm) * 1e-3)
    ax2.plot([gg], [kk], "o", color=SERIES[1]); ax2.annotate(f"{kk:.3f}", (gg, kk), xytext=(6, 6), textcoords="offset points", fontsize=8)
ax2.set_xlabel("ring–bus gap (nm)"); ax2.set_ylabel("κ²")
ax2.set_title("Reference ring: κ² = 0.107 at a 200 nm gap;\n±10 nm of gap moves κ² across the critical-coupling line", fontsize=10)
ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "capstone_gap_sensitivity.png"); plt.close(fig)


# ============================================================================= PART D
# The same phasor seen two more ways.
#  (1) The transverse factor e^{-j k_x x} drawn in the complex plane as x increases (notes 17):
#      real k_x -> the phasor rotates on the unit circle; k_x = -j*gamma -> it walks straight to the origin.
#  (2) The time-averaged Poynting vector <S> = 1/2 Re{E x H*} computed numerically from the phasor
#      (notes 18 for H via Faraday, notes 25 for <S>): along k when k_x is real; when k_x = -j*gamma the
#      normal component <S_x> is exactly zero (reactive) while <S_z> > 0 slides along the interface.
ETA0 = MU0 * C0                       # free-space impedance, 376.73 ohm


def phasor_fields(X_um, Z_um, kx, kz):
    """E_y, H_x, H_z phasors in SI for E = y E0 e^{-j(k_x x + k_z z)}, E0 = 1 V/m.
    Faraday: curl E = -j w mu H with E = y E_y gives H_x = -(k_z/w mu) E_y and H_z = (k_x/w mu) E_y (notes 18)."""
    kx_m, kz_m = kx * 1e6, kz * 1e6                      # rad/um -> rad/m
    Ey = np.exp(-1j * (kx_m * X_um * 1e-6 + kz_m * Z_um * 1e-6))
    Hx = -(kz_m / (omega * MU0)) * Ey
    Hz = (kx_m / (omega * MU0)) * Ey
    return Ey, Hx, Hz


def poynting_avg(Ey, Hx, Hz):
    """<S> = 1/2 Re{E x H*} for E = y E_y: S_x = 1/2 Re(E_y H_z*), S_z = -1/2 Re(E_y H_x*), in W/m^2 (notes 25).
    Also returns the reactive (imaginary) normal part 1/2 Im(E_y H_z*), which is what survives when k_x is imaginary."""
    Sx = 0.5 * np.real(Ey * np.conj(Hz))
    Sz = -0.5 * np.real(Ey * np.conj(Hx))
    Sx_reactive = 0.5 * np.imag(Ey * np.conj(Hz))
    return Sx, Sz, Sx_reactive


# (1) phasor walk numbers
xs_D = np.linspace(0, 1.0, 400)
kx_prop = kx_of(neff_prop * k0)                          # n_eff = neff_prop (1.0 < n2): k_x real
T_real = np.exp(-1j * kx_prop * xs_D)
T_imag = np.exp(-1j * kx_ref * xs_D)                     # n_eff = neff_ref (2.5): k_x = -j gamma
RESULTS["partD_propagating_example_neff"] = neff_prop
RESULTS["partD_propagating_example_kx_rad_per_um"] = float(kx_prop)
RESULTS["partD_propagating_example_phase_per_100nm_rad"] = float(-kx_prop * 0.1)
RESULTS["partD_phasor_real_kx_max_abs_dev_from_unit_circle"] = float(np.max(np.abs(np.abs(T_real) - 1)))
RESULTS["partD_phasor_imag_kx_max_abs_phase_rad"] = float(np.max(np.abs(np.angle(T_imag))))
RESULTS["partD_phasor_imag_kx_value_at_1_over_gamma"] = float(np.exp(-1j * kx_ref * (1 / gamma_ref)).real)   # e^-1
print(f"Part D: |e^(-j kx x)| - 1 (real kx) max = {RESULTS['partD_phasor_real_kx_max_abs_dev_from_unit_circle']:.1e}; "
      f"phase of e^(-j kx x) (kx = -j gamma) max = {RESULTS['partD_phasor_imag_kx_max_abs_phase_rad']:.1e} rad; "
      f"value at x = 1/gamma = {RESULTS['partD_phasor_imag_kx_value_at_1_over_gamma']:.4f} (e^-1 = {math.exp(-1):.4f})")

# (2) Poynting vector, propagating case (the Part A tilted wave in silica).
# The numbers are evaluated on a coarse grid (the same one the propagating quiver uses).
zq = np.linspace(0.1, 1.5, 12); xq = np.linspace(0.05, 0.55, 6)
ZQ, XQ = np.meshgrid(zq, xq)
Ey_p, Hx_p, Hz_p = phasor_fields(XQ, ZQ, kx_A, kz_A)
Sx_p, Sz_p, _ = poynting_avg(Ey_p, Hx_p, Hz_p)
S_angle_p = math.degrees(math.atan2(Sx_p.mean(), Sz_p.mean()))
S_mag_p = float(np.hypot(Sx_p, Sz_p).mean())
RESULTS["partD_S_angle_deg_tilted_wave"] = S_angle_p
RESULTS["partD_S_angle_expected_deg"] = theta_deg
RESULTS["partD_S_angle_agreement_pct"] = 100 * S_angle_p / theta_deg
RESULTS["partD_S_magnitude_W_per_m2_tilted_wave_E0_1V_per_m"] = S_mag_p
RESULTS["partD_S_magnitude_expected_n2_over_2eta0"] = n2 / (2 * ETA0)
RESULTS["partD_S_magnitude_agreement_pct"] = 100 * S_mag_p / (n2 / (2 * ETA0))
# evanescent case (n_eff = 2.5)
Ey_e, Hx_e, Hz_e = phasor_fields(XQ, ZQ, kx_ref, kz_ref)
Sx_e, Sz_e, Sx_e_react = poynting_avg(Ey_e, Hx_e, Hz_e)
RESULTS["partD_evanescent_max_abs_Sx_over_Sz"] = float(np.max(np.abs(Sx_e) / Sz_e))
RESULTS["partD_evanescent_Sx_expected"] = 0.0
RESULTS["partD_evanescent_reactive_Sx_over_Sz"] = float(np.mean(Sx_e_react / Sz_e))       # = gamma/beta
RESULTS["partD_evanescent_reactive_expected_gamma_over_beta"] = gamma_ref / kz_ref
Ey0, Hx0, Hz0 = phasor_fields(np.array([0.0]), np.array([0.0]), kx_ref, kz_ref)
Sz0 = poynting_avg(Ey0, Hx0, Hz0)[1][0]
RESULTS["partD_evanescent_Sz_at_interface_W_per_m2_E0_1V_per_m"] = float(Sz0)
RESULTS["partD_evanescent_Sz_expected_neff_over_2eta0"] = neff_ref / (2 * ETA0)
RESULTS["partD_evanescent_Sz_agreement_pct"] = 100 * float(Sz0) / (neff_ref / (2 * ETA0))
RESULTS["partD_evanescent_Sz_decay_length_nm"] = 1e3 / (2 * gamma_ref)          # power decays as e^{-2 gamma x}
print(f"  tilted wave: <S> at {S_angle_p:.2f} deg (theta = {theta_deg}), |<S>| = {S_mag_p:.4e} W/m^2 "
      f"(n2/2eta0 = {n2/(2*ETA0):.4e}); evanescent: max|Sx/Sz| = {RESULTS['partD_evanescent_max_abs_Sx_over_Sz']:.1e}, "
      f"reactive Sx/Sz = {RESULTS['partD_evanescent_reactive_Sx_over_Sz']:.3f} (gamma/beta = {gamma_ref/kz_ref:.3f}), "
      f"Sz(0) = {Sz0:.4e} W/m^2 (n_eff/2eta0 = {neff_ref/(2*ETA0):.4e})")

# ---- figure D
fig = plt.figure(figsize=(13, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], hspace=0.55, wspace=0.18, left=0.05, right=0.99, top=0.84, bottom=0.09)
axPh = fig.add_subplot(gs[:, 0]); axSp = fig.add_subplot(gs[0, 1]); axSe = fig.add_subplot(gs[1, 1])
marks = np.arange(0, 1.0001, 0.1)
circ = np.exp(1j * np.linspace(0, 2 * math.pi, 200))
axPh.plot(circ.real, circ.imag, color=PALETTE["line"], lw=1, ls="--")
axPh.plot(T_real.real, T_real.imag, color=SERIES[0], lw=2, label=f"n_eff = {neff_prop:.1f} < n₂: k_x = {kx_prop:.2f} rad/µm (real)")
Tm = np.exp(-1j * kx_prop * marks); axPh.plot(Tm.real, Tm.imag, "o", color=SERIES[0], ms=5)
axPh.plot(T_imag.real, T_imag.imag, color=SERIES[1], lw=3, label=f"n_eff = {neff_ref} > n₂: k_x = −j{gamma_ref:.2f}/µm (imaginary)")
Tm = np.exp(-1j * kx_ref * marks); axPh.plot(Tm.real, Tm.imag, "o", color=SERIES[1], ms=5, zorder=5)
axPh.annotate("x = 0", (1.0, 0.0), xytext=(8, -14), textcoords="offset points", fontsize=8)
axPh.annotate(f"x = 1/γ = {decay_ref_nm:.0f} nm:\ne⁻¹ = 0.368", (math.exp(-1), 0.0), xytext=(-10, 28), textcoords="offset points",
              fontsize=8, color=SERIES[1], arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.8))
p100 = np.exp(-1j * kx_prop * 0.1)
# label placed inside the circle, well clear of the blue curve (the leader runs inward from the dot)
axPh.annotate(f"x = 100 nm:\nphase {-kx_prop*0.1:.2f} rad", (p100.real, p100.imag), xytext=(0.5, -0.62), textcoords="data",
              ha="right", va="center", fontsize=8, color=SERIES[0], arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.8))
axPh.set_aspect("equal"); axPh.set_xlim(-1.25, 1.25); axPh.set_ylim(-1.25, 1.25)
axPh.axhline(0, color=PALETTE["line"], lw=0.8); axPh.axvline(0, color=PALETTE["line"], lw=0.8)
axPh.set_xlabel("Re{e^{−jk_x x}}"); axPh.set_ylabel("Im{e^{−jk_x x}}")
axPh.set_title("The transverse factor as a phasor, one dot per 100 nm of x:\nreal k_x rotates it, imaginary k_x shrinks it", fontsize=9)
axPh.legend(loc="lower left", fontsize=7.5, facecolor=PALETTE["surface"], framealpha=0.9, frameon=True)

# Propagating panel: the same grid as the numbers, one global scale (|<S>| is uniform, so nothing is lost).
# Evanescent panel: <S_z> falls as e^{-2 gamma x} with 1/(2 gamma) = 51 nm, so rows 100 nm apart on one global
# scale would leave everything above the first row invisible. Rows are packed near the interface, each row's arrows
# are normalised to that row's own |<S>| (direction is what the panel is about), and the true ratio
# <S_z>(x)/<S_z>(0) = e^{-2 gamma x} is printed at the end of every row; the |E|^2 background carries the decay.
xq_e = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.45])      # first row at 50 nm: a 20 nm row would sit on top of it
zq_e = np.linspace(0.1, 1.2, 10)
ZQe, XQe = np.meshgrid(zq_e, xq_e)
Sx_e_plot, Sz_e_plot, _ = poynting_avg(*phasor_fields(XQe, ZQe, kx_ref, kz_ref))
Sz_e_ratio = Sz_e_plot[:, 0] / Sz0                           # = e^{-2 gamma x}, one value per row
RESULTS["partD_evanescent_Sz_ratio_rows_x_um"] = [float(v) for v in xq_e]
RESULTS["partD_evanescent_Sz_ratio_rows"] = [float(v) for v in Sz_e_ratio]
RESULTS["partD_evanescent_Sz_ratio_rows_expected"] = [float(math.exp(-2 * gamma_ref * xv)) for xv in xq_e]

for ax_, (Sx_, Sz_, ZQ_, XQ_, kx_, kz_, per_row, vmax_, ttl) in zip(
        (axSp, axSe),
        ((Sx_p, Sz_p, ZQ, XQ, kx_A, kz_A, False, 2.0,
          f"Propagating wave (n_eff = k_z/k₀ = {kz_A/k0:.2f} < n₂): ⟨S⟩ points along k,\n{S_angle_p:.1f}° from z (θ = {theta_deg:.0f}°), |⟨S⟩| = n₂|E₀|²/2η₀ everywhere"),
         (Sx_e_plot, Sz_e_plot, ZQe, XQe, kx_ref, kz_ref, True, 1.0,
          f"Evanescent (n_eff = {neff_ref}): ⟨S_x⟩ = 0 exactly, ⟨S_z⟩ ∝ e^{{−2γx}} (1/2γ = {1e3/(2*gamma_ref):.0f} nm): power slides along\nthe interface, none crosses it. Arrows normalised per row; ⟨S_z⟩/⟨S_z⟩(0) printed at each row's end"))):
    xb = np.linspace(0, 0.6, 240); zb = np.linspace(0, 1.6, 320); ZBB, XBB = np.meshgrid(zb, xb)
    Ey_b = phasor_fields(XBB, ZBB, kx_, kz_)[0]
    # propagating: vmax = 2 keeps the uniform |E|^2 = 1 a mid blue so the arrows stay readable; evanescent: vmax = 1 = |E|^2 at x = 0
    ax_.imshow(np.abs(Ey_b) ** 2, extent=[zb[0], zb[-1], xb[0], xb[-1]], origin="lower", cmap="Blues", vmin=0, vmax=vmax_, aspect="equal")
    if per_row:
        srow = np.hypot(Sx_, Sz_).max(axis=1, keepdims=True)
        U, V = Sz_ / srow, Sx_ / srow
        for xv, r in zip(xq_e, Sz_e_ratio):
            ax_.text(zq_e[-1] + 0.1, xv, f"×{r:.2g}" if r >= 1e-3 else f"×{r:.0e}", fontsize=7, va="center", ha="left", color=PALETTE["ink"])
    else:
        smax = float(np.hypot(Sx_, Sz_).max())
        U, V = Sz_ / smax, Sx_ / smax
    ax_.quiver(ZQ_, XQ_, U, V, angles="xy", scale_units="xy", scale=1 / 0.11, color=PALETTE["ink"], width=0.006, zorder=5)
    ax_.axhline(0, color=PALETTE["ink"], lw=1.5); ax_.grid(False)
    ax_.set_xlim(zb[0], zb[-1]); ax_.set_ylim(xb[0], xb[-1])
    ax_.set_xlabel("z (µm), along the interface"); ax_.set_ylabel("x (µm)")
    ax_.set_title(ttl, fontsize=9)
fig.suptitle("Same phasor E = E₀e^{−j(k_x x + k_z z)}, two more views: the transverse factor in the complex plane (left)\n"
             "and the time-averaged power flow ⟨S⟩ = ½Re{E×H*} (right, arrows; background |E|² in Blues, silica side x > 0)", fontsize=10.5)
fig.savefig(OUT / "transverse_phasor_and_power.png"); plt.close(fig)
print("Part D figure written")

ffmpeg_version = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.split()[2]

# ============================================================================= write outputs
RESULTS["runtime_s"] = time.time() - T_START
with open(OUT / "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

tools = [
    {
        "tool": "numpy",
        "version": np.__version__,
        "what_it_is": "The array library under all of scientific Python: vectorised arithmetic on grids, complex numbers, FFTs, linear algebra.",
        "used_for": "Evaluating E = Re{exp(j(ωt − k_x x − k_z z))} on a 2-D grid with k_x either real or purely imaginary (k_x = −jγ), measuring wavelengths along z, x and k from zero crossings, checking the identity Re{e^{j(ωt − k_x x − βz)}} = e^{−γx}cos(ωt − βz) to machine precision, drawing the transverse factor e^{−jk_x x} in the complex plane, and building the E and H phasors (Faraday's law) to evaluate the time-averaged Poynting vector ⟨S⟩ = ½Re{E×H*} numerically.",
        "result": f"λ in silica {RESULTS['lambda_in_silica_nm']:.1f} nm (λ0/n2 = {lam0_nm/n2:.1f}); at θ = 30° measured 2π/k_z = {lam_z_meas*1e3:.1f} nm vs {lam_z_um*1e3:.1f} analytic and 2π/k_x = {lam_x_meas*1e3:.1f} vs {lam_x_um*1e3:.1f}; identity check max difference {RESULTS['identity_check_max_abs_diff']:.1e}; k_x² + k_z² = {RESULTS['dispersion_check_kx2_plus_kz2_over_k02']:.4f} k0² = n2² exactly; ⟨S⟩ of the tilted wave at {S_angle_p:.2f}° (θ = {theta_deg:.0f}°) with |⟨S⟩| = n2/2η0 to {RESULTS['partD_S_magnitude_agreement_pct']:.1f} %; evanescent ⟨S_x⟩/⟨S_z⟩ = {RESULTS['partD_evanescent_max_abs_Sx_over_Sz']:.0e} (expected 0), reactive part γ/β = {RESULTS['partD_evanescent_reactive_Sx_over_Sz']:.3f}.",
        "how_to_observe": "cd experiments/02_wavevector_evanescent && ../../.venv/bin/python run.py ; numbers print to the terminal and go to out/results.json. Change theta_deg (Part A) or neff_ref / n2 in run.py and re-run.",
    },
    {
        "tool": "scipy",
        "version": scipy.__version__,
        "what_it_is": "Scientific algorithms on top of numpy: optimisation, curve fitting, integration, signal processing, special functions.",
        "used_for": "scipy.optimize.curve_fit fits A·e^{−x/L} to the envelope of the computed evanescent field to recover the decay length from the field itself, independently of the formula 1/γ.",
        "result": f"Fitted 1/γ = {RESULTS['decay_length_fitted_from_field_nm']:.2f} nm vs analytic {decay_ref_nm:.2f} nm ({RESULTS['decay_length_fit_agreement_pct']:.3f} %).",
        "how_to_observe": f"out/results.json keys decay_length_fitted_from_field_nm and decay_length_nm_neff{neff_tag}_silica; the envelope is the dashed curve in the lower-right panel of out/evanescent_sweep.mp4.",
    },
    {
        "tool": "matplotlib (FuncAnimation + FFMpegWriter)",
        "version": matplotlib.__version__,
        "what_it_is": "The standard Python plotting library; its animation module re-draws a figure frame by frame and pipes the frames to ffmpeg to make a video.",
        "used_for": "Signed-field colour maps (RdBu_r centred on zero) of the plane wave and of the evanescent field; the two videos plane_wave_tilted.mp4 (time running, crest tracking) and evanescent_sweep.mp4 (k_z swept through n2·k0 while the k_x² parabola and the x-profile update); the five static figures (including the phasor-plane and quiver plot of ⟨S⟩) and two contact sheets.",
        "result": "out/plane_wave_tilted.png/.mp4/_frames.png, out/evanescent_sweep.mp4/_frames.png, out/transverse_phasor_and_power.png, out/decay_length_vs_neff.png, out/tail_log_scale.png, out/capstone_gap_sensitivity.png.",
        "how_to_observe": "Open the mp4s in QuickTime/VLC. In plane_wave_tilted.mp4 watch the circle (crest along k) and the square (same crest on the z axis): the square is faster. In evanescent_sweep.mp4 watch the dot on the parabola cross zero at n_eff = 1.45 while the field pattern stops oscillating in x. Edit neff_path or n_frames_A in run.py to change the sweep range or the duration.",
    },
    {
        "tool": "ffmpeg",
        "version": f"{ffmpeg_version} ({FFMPEG})",
        "what_it_is": "The universal command-line video encoder/decoder; matplotlib calls it to turn a sequence of frames into an H.264 mp4.",
        "used_for": "Encoding both animations at 30 fps with libx264 and yuv420p pixel format so they play in any viewer.",
        "result": f"out/plane_wave_tilted.mp4 ({n_frames_A} frames, {n_frames_A/30:.0f} s) and out/evanescent_sweep.mp4 ({n_frames_B} frames, {n_frames_B/30:.1f} s).",
        "how_to_observe": "/opt/homebrew/bin/ffprobe out/evanescent_sweep.mp4 prints duration, resolution and codec.",
    },
]
with open(OUT / "tools.json", "w") as f:
    json.dump(tools, f, indent=2)

print(f"done in {RESULTS['runtime_s']:.1f} s; outputs in {OUT}")
