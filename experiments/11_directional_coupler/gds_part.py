"""gdsfactory part of experiment 11 (run with the .venv interpreter; run.py calls it via subprocess).

Draws the two coupler geometries of the capstone at mask level and quantifies the ring-bus
"point coupler":
  * ring_single: ring R = 6.3 µm (circular bends), 500 nm strips, 200 nm gap to a straight bus
    -> out/ring_bus_coupler.gds and a matplotlib drawing out/gds_layout.png (left panel)
  * coupler: a straight directional coupler, 200 nm gap, length = L_c of the 200 nm gap from the
    analytic supermodes (slab_coupler.py) -> out/directional_coupler.gds (right panel)
  * point coupler: along the bus the ring-bus separation is gap(z) = g0 + R - sqrt(R² - z²)
    ≈ g0 + z²/(2R).  With κ(z) = κ0 e^{-γ (gap(z) - g0)} the accumulated coupling is
    ∫κ dz = κ0 · sqrt(2πR/γ) = κ0 · L_eff : a ring touching a bus acts like a straight coupler of
    length L_eff (~2 µm) at the minimum gap.  Checked numerically against the exact circle.

The gap and the ring radius are measured back from the GDS polygons (a layout check), and the
z²/(2R) parabola is compared with the ring's centreline circle (polygon vertices scaled radially
to R), which is what the mode-centre separation and the L_eff integral use.  (Comparing it with
the outer edge would mix in the w/2 radius offset: the edge circle has radius R + w/2.)
Writes out/gds_results.json.
"""
import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import gdsfactory as gf
from common import REF, use_style, SERIES, PALETTE
from common.params import k0_per_um
import slab_coupler as sc

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
use_style()
LAM = REF.lambda_nm * 1e-3
K0 = k0_per_um(REF.lambda_nm)
R, GAP, W = REF.radius_um, 0.200, REF.wg_width_um
gf.gpdk.PDK.activate()
res = {"versions": {"gdsfactory": gf.__version__}}


def polygons(comp):
    """All polygons of a component (layer (1,0) = silicon) as a list of (N,2) µm arrays."""
    d = comp.get_polygons_points(by="tuple")
    return [np.asarray(p) for polys in d.values() for p in polys]


def draw(ax, polys, color=SERIES[0]):
    for p in polys:
        ax.add_patch(MplPolygon(p, closed=True, facecolor=color, edgecolor=PALETTE["ink2"], lw=0.4, alpha=0.85))
    allp = np.vstack(polys)
    ax.set_xlim(allp[:, 0].min() - 0.5, allp[:, 0].max() + 0.5)
    ax.set_ylim(allp[:, 1].min() - 0.5, allp[:, 1].max() + 0.5)
    ax.set_aspect("equal")


# ----------------------------------------------------------------------------- ring + bus (point coupler)
ring = gf.components.ring_single(radius=R, gap=GAP, length_x=0.01, length_y=0.01, bend="bend_circular",
                                 cross_section="strip")
ring.write_gds(OUT / "ring_bus_coupler.gds")
rp = polygons(ring)
allpts = np.vstack(rp)
bus = [p for p in rp if p[:, 1].max() < 0.6]            # the bus strip is the only polygon below y ~ 0.5
bus_top = max(p[:, 1].max() for p in bus)
ring_polys = [p for p in rp if p[:, 1].max() >= 0.6]
ring_pts = np.vstack(ring_polys)
ring_bottom = ring_pts[:, 1].min()
gap_measured = ring_bottom - bus_top
# ring centre and radii from the ring polygon vertices (outer/inner circle)
xc = 0.5 * (ring_pts[:, 0].min() + ring_pts[:, 0].max())
yc = 0.5 * (ring_pts[:, 1].min() + ring_pts[:, 1].max())
rr = np.hypot(ring_pts[:, 0] - xc, ring_pts[:, 1] - yc)
r_outer, r_inner = rr.max(), rr.min()
res["ring_bus"] = dict(gds=str(OUT / "ring_bus_coupler.gds"), n_polygons=len(rp),
                       gap_nm_measured=float(gap_measured * 1e3), gap_nm_requested=GAP * 1e3,
                       radius_um_centreline=float(0.5 * (r_outer + r_inner)), width_um=float(r_outer - r_inner),
                       bbox_um=[float(v) for v in (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max())])
print(f"[gds] ring_single: {len(rp)} polygons; measured gap {gap_measured*1e3:.1f} nm, centreline R {0.5*(r_outer+r_inner):.3f} µm, "
      f"width {r_outer-r_inner:.3f} µm")

# ----------------------------------------------------------------------------- straight directional coupler
Lc200 = sc.supermodes(0.20, lam=LAM)["L_c"]
dc = gf.components.coupler(gap=GAP, length=float(round(Lc200, 2)), dy=3.0, dx=6.0, cross_section="strip")
dc.write_gds(OUT / "directional_coupler.gds")
dp = polygons(dc)
res["directional_coupler"] = dict(gds=str(OUT / "directional_coupler.gds"), n_polygons=len(dp), gap_nm=GAP * 1e3,
                                  straight_length_um=float(round(Lc200, 2)),
                                  ports={p.name: [float(v) for v in p.dcenter] for p in dc.ports})
print(f"[gds] coupler: straight section {Lc200:.2f} µm (= L_c of the 200 nm gap), ports {list(res['directional_coupler']['ports'])}")

fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.0, 1.4]))
ax = axs[0]; draw(ax, rp)
ax.annotate("", xy=(xc + R * 0.0, yc), xytext=(xc + R * np.cos(np.radians(35)), yc + R * np.sin(np.radians(35))),
            arrowprops=dict(arrowstyle="<->", color=PALETTE["ink2"], lw=1))
ax.text(xc + 1.0, yc + 1.5, f"R = {R} µm", fontsize=8)
ax.annotate(f"gap {GAP*1e3:.0f} nm\n(measured {gap_measured*1e3:.0f} nm)", xy=(xc, ring_bottom), xytext=(xc + 3.5, ring_bottom - 1.6),
            fontsize=8, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"], lw=0.8))
ax.text(allpts[:, 0].min() + 0.3, bus_top - 1.1, "bus, 500 nm strip", fontsize=8)
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
ax.set_title("Ring-bus point coupler (gdsfactory ring_single):\nthe coupling happens where the circle grazes the bus", fontsize=9.5)
ax = axs[1]; draw(ax, dp)
allpd = np.vstack(dp)
for p in dc.ports:
    x, y = p.dcenter
    ax.plot(x, y, "o", color=SERIES[3], ms=4); ax.text(x + (0.3 if x > 0 else -1.2), y + 0.25, p.name, fontsize=8)
ax.annotate("", xy=(0, -2.1), xytext=(round(Lc200, 2), -2.1), arrowprops=dict(arrowstyle="<->", color=PALETTE["ink2"], lw=1))
ax.text(Lc200 / 2, -2.6, f"parallel section L = L_c = {Lc200:.2f} µm (full transfer at 200 nm gap)", ha="center", fontsize=8)
ax.set_ylim(allpd[:, 1].min() - 1.4, allpd[:, 1].max() + 0.5)
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
ax.set_title("Straight directional coupler (gdsfactory coupler): two 500 nm strips,\n200 nm gap, S-bends bring them in and out", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "gds_layout.png"); plt.close(fig)

# ----------------------------------------------------------------------------- point-coupler effective length
z = np.linspace(-R, R, 20001)
gap_exact = GAP + R - np.sqrt(np.maximum(R ** 2 - z ** 2, 0.0))          # circle grazing the bus
gap_parab = GAP + z ** 2 / (2 * R)
# the polygon edge: outer-circle vertices of the ring polygons near the bottom (radius R + w/2)
edge = ring_pts[(ring_pts[:, 1] < yc) & (np.abs(rr - r_outer) < 2e-3)]
edge = edge[np.argsort(edge[:, 0])]
edge_gap = edge[:, 1] - bus_top
edge_z = edge[:, 0] - xc
# the ring CENTRELINE circle (radius R = (r_outer + r_inner)/2): every lower vertex scaled radially onto R.
# The mode centres sit on this circle, so this, not the outer edge, is what gap(z) = g0 + z²/(2R) describes.
R_c = 0.5 * (r_outer + r_inner)
low = ring_pts[ring_pts[:, 1] < yc]
cl = np.column_stack([xc + (low[:, 0] - xc) * R_c / np.hypot(low[:, 0] - xc, low[:, 1] - yc),
                      yc + (low[:, 1] - yc) * R_c / np.hypot(low[:, 0] - xc, low[:, 1] - yc)])
cl = cl[np.argsort(cl[:, 0])]
cl_z = cl[:, 0] - xc
cl_rise = cl[:, 1] - (yc - R_c)                      # centreline height above its lowest point = separation increase
bus_centre = bus_top - W / 2
pc = {}
g_slab = sc.single_slab_te(lam=LAM)["gamma"]
g_text = K0 * np.sqrt(REF.neff ** 2 - REF.n_sio2 ** 2)
for name, gam in (("textbook_neff2p5", g_text), ("slab2d_neff2p99", g_slab)):
    kz_exact = np.exp(-gam * (gap_exact - GAP))
    kz_parab = np.exp(-gam * (gap_parab - GAP))
    L_num = float(np.trapezoid(kz_exact, z)); L_par = float(np.trapezoid(kz_parab, z))
    L_formula = float(sc.point_coupler_length(R, gam))
    pc[name] = dict(gamma_per_um=float(gam), decay_length_nm=float(1e3 / gam), L_eff_formula_um=L_formula,
                    L_eff_numeric_exact_circle_um=L_num, L_eff_numeric_parabola_um=L_par,
                    agreement_pct=float(100 * L_num / L_formula), z_1_over_e_um=float(np.sqrt(2 * R / gam)),
                    gap_at_z_1_over_e_nm=float(1e3 * (GAP + 1 / gam)))
    print(f"[gds] point coupler ({name}): γ = {gam:.2f}/µm -> L_eff = √(2πR/γ) = {L_formula:.3f} µm; "
          f"numeric (exact circle) {L_num:.3f} µm ({100*L_num/L_formula:.2f} %)")
res["point_coupler"] = pc
# parabola check against the centreline circle: expected deviation of z²/(2R) from R − √(R² − z²) is ≈ z⁴/(8R³)
sel25 = np.abs(cl_z) < 2.5
dev_cl = 1e3 * (np.interp(cl_z, z, z ** 2 / (2 * R)) - cl_rise)
z_1e = float(np.sqrt(2 * R / g_text))
res["parabola_check"] = dict(
    compared_with="ring centreline circle (GDS lower vertices scaled radially to R), i.e. the mode-centre separation",
    centreline_radius_um=float(R_c),
    max_dev_nm_within_2p5um=float(np.max(np.abs(dev_cl[sel25]))),
    z_of_max_dev_um=float(np.abs(cl_z[sel25])[np.argmax(np.abs(dev_cl[sel25]))]),
    expected_z4_over_8R3_at_that_z_nm=float(1e3 * np.abs(cl_z[sel25])[np.argmax(np.abs(dev_cl[sel25]))] ** 4 / (8 * R ** 3)),
    expected_z4_over_8R3_at_2p5um_nm=float(1e3 * 2.5 ** 4 / (8 * R ** 3)),
    max_dev_nm_within_z_1_over_e=float(np.max(np.abs(dev_cl[np.abs(cl_z) < z_1e]))),
    z_1_over_e_um=z_1e, expected_z4_over_8R3_at_z_1_over_e_nm=float(1e3 * z_1e ** 4 / (8 * R ** 3)),
    note="the outer polygon edge (radius R + w/2) happens to lie within 7 nm of the R-parabola out to 2.5 µm; that is a "
         "coincidence of the two radii, not a validation, so the check is done on the centreline",
    outer_edge_vs_R_parabola_max_dev_nm_within_2p5um=float(1e3 * np.max(np.abs(np.interp(edge_z, z, gap_parab) - edge_gap)[np.abs(edge_z) < 2.5])))
print(f"[gds] parabola z²/(2R) vs centreline circle: max |dev| = {res['parabola_check']['max_dev_nm_within_2p5um']:.1f} nm at |z| = "
      f"{res['parabola_check']['z_of_max_dev_um']:.2f} µm (expected z⁴/8R³ = {res['parabola_check']['expected_z4_over_8R3_at_that_z_nm']:.1f} nm there), "
      f"{res['parabola_check']['max_dev_nm_within_z_1_over_e']:.2f} nm inside |z| < √(2R/γ) = {z_1e:.2f} µm")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axs[0]
draw(ax, rp); ax.set_xlim(xc - 3.2, xc + 3.2); ax.set_ylim(bus_top - 0.9, bus_top + 1.6); ax.set_aspect("auto")   # zoom: y stretched ~2.6x
ax.plot(xc + z, yc - R + z ** 2 / (2 * R), color=SERIES[1], lw=1.4, label="ring centreline: y(0) + z²/(2R)  (parabola)")
ax.plot(cl[:, 0], cl[:, 1], "o", ms=2.5, color=SERIES[4], label="ring centreline from the GDS vertices (exact circle, radius R)")
ax.axhline(bus_centre, color=SERIES[4], lw=0.9, ls="--", label="bus centreline")
ax.plot(xc + edge_z, bus_top + edge_gap, ".", ms=2, color=PALETTE["ink2"], alpha=0.5, label="outer polygon edge (radius R + w/2, for reference)")
for gam, col, lab in ((g_text, SERIES[2], "textbook γ"), (g_slab, SERIES[3], "slab γ")):
    ze = np.sqrt(2 * R / gam)
    ax.axvspan(xc - ze, xc + ze, color=col, alpha=0.12, label=f"|z| < √(2R/γ) = {ze:.2f} µm: κ within 1/e ({lab})")
ax.set_xlabel("z along the bus (µm)"); ax.set_ylabel("y (µm)  (vertical scale stretched)")
ax.set_title("Zoom on the touching point: the ring-bus centre separation\ngrows as z²/(2R) over the ~2 µm that matter", fontsize=9.5)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)   # below the x-axis label
ax = axs[1]
for i, (name, gam, lab) in enumerate((("textbook_neff2p5", g_text, "1/γ = 102 nm (n_eff 2.5, textbook)"),
                                      ("slab2d_neff2p99", g_slab, "1/γ = 80 nm (2-D slab, n_eff 2.99)"))):
    kz = np.exp(-gam * (gap_exact - GAP)); Le = pc[name]["L_eff_formula_um"]
    ax.plot(z, kz, color=SERIES[i], label=f"κ(z)/κ₀ = e^(−γ z²/2R), {lab}")
    ax.fill_between([-Le / 2, Le / 2], 0, 1, color=SERIES[i], alpha=0.12, label=f"same area: rectangle L_eff = {Le:.2f} µm")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(0, 1.05)
ax.set_xlabel("z along the bus (µm)"); ax.set_ylabel("local coupling κ(z) / κ₀")
ax.set_title("A ring grazing a bus = a straight coupler of length L_eff = √(2πR/γ)\nat the minimum gap (Gaussian and rectangle have equal area)", fontsize=9.5)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "point_coupler_geometry.png", bbox_inches="tight"); plt.close(fig)

res["figures"] = ["gds_layout.png", "point_coupler_geometry.png"]
(OUT / "gds_results.json").write_text(json.dumps(res, indent=2))
print("[gds] wrote", OUT / "gds_results.json")
