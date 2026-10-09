"""Full 3-D FDTD of the 150 nm-gap coupler with Meep: the independent check.

Nothing from coupled-mode theory goes in. Meep steps Maxwell's equations in time
on a Yee grid for the real 3-D structure (two 500 x 220 nm silicon strips in
silica), with guide B starting abruptly at z = 0 so that the launch is exactly
A(0) = 1, B(0) = 0. Flux planes every 1 um measure the power in each half of the
cross-section, and Meep's built-in eigenmode solver (MPB) gives the supermode
indices on the same grid.

Run (about 20 to 25 min, single process; MPI was slower on this Mac):
    cd ~/photonics-simulations/visuals/03_coupled_modes && ../../.meep/bin/python fdtd3d.py
Writes out/fdtd/fdtd3d_res<RES>.npz and .json; run.py uses the finest resolution present
and lists the others as a convergence check. Optional argument: resolution (default 20).

Axes: Meep x = propagation (the notes' z), Meep y = across the chip (the notes'
x), Meep z = vertical (the notes' y). Everything saved is relabelled to the notes'
axes.
"""

import json
import sys
import time
from pathlib import Path

import meep as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import indices as ix  # noqa: E402

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 20     # pixels per um
LAM, W, H, GAP = 1.31, 0.50, 0.22, 0.15
L = float(sys.argv[2]) if len(sys.argv) > 2 else 72.0              # length of the coupled section (one transfer is ~65 um)
LEAD = 3.0            # guide A alone before B starts
DPML = 0.8
CLAD_Y, CLAD_Z = 0.7, 0.6
F0 = 1 / LAM
N_SI, N_OX = float(ix.n_silicon(LAM)), float(ix.n_silica(LAM))

OUT = Path(__file__).resolve().parent / "out" / "fdtd"
OUT.mkdir(parents=True, exist_ok=True)

c = (W + GAP) / 2
sx = LEAD + L + 2 * DPML
sy = 2 * c + W + 2 * CLAD_Y + 2 * DPML
sz = H + 2 * CLAD_Z + 2 * DPML
x0 = -sx / 2 + DPML + LEAD          # where guide B starts: the notes' z = 0
si, ox = mp.Medium(index=N_SI), mp.Medium(index=N_OX)

geometry = [
    mp.Block(mp.Vector3(mp.inf, W, H), center=mp.Vector3(0, -c, 0), material=si),          # A
    mp.Block(mp.Vector3(sx, W, H), center=mp.Vector3(x0 + sx / 2, c, 0), material=si),     # B
]
src_x = x0 - 2.0
sources = [mp.EigenModeSource(
    mp.GaussianSource(F0, fwidth=0.2 * F0),
    center=mp.Vector3(src_x, -c, 0), size=mp.Vector3(0, W + 0.8, H + 0.8),
    eig_band=1, eig_parity=mp.EVEN_Z, direction=mp.X, eig_kpoint=mp.Vector3(F0 * 2.7))]

sim = mp.Simulation(
    cell_size=mp.Vector3(sx, sy, sz), resolution=RES, geometry=geometry,
    default_material=ox, boundary_layers=[mp.PML(DPML)], sources=sources,
    symmetries=[mp.Mirror(mp.Z, phase=+1)],      # quasi-TE: Ex, Ey even about mid-height
)

ymax, zmax = sy / 2 - DPML, sz / 2 - DPML
zs = np.arange(-1.0, L - 0.5, 1.0)              # the notes' z of each flux plane
flux_a, flux_b = [], []
for z in zs:
    x = x0 + z
    flux_a.append(sim.add_flux(F0, 0, 1, mp.FluxRegion(
        center=mp.Vector3(x, -ymax / 2, 0), size=mp.Vector3(0, ymax, 2 * zmax))))
    flux_b.append(sim.add_flux(F0, 0, 1, mp.FluxRegion(
        center=mp.Vector3(x, ymax / 2, 0), size=mp.Vector3(0, ymax, 2 * zmax))))

top = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], F0, 0, 1,
                         center=mp.Vector3(x0 + L / 2 - 1.0, 0, 0),
                         size=mp.Vector3(L + 2.0, 2 * ymax, 0))
xsec_z = [z for z in (0.0, 16.0, 32.0, 48.0, 64.0) if z < L]
xsec = [sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], F0, 0, 1,
                           center=mp.Vector3(x0 + z, 0, 0), size=mp.Vector3(0, 2 * ymax, 2 * zmax))
        for z in xsec_z]

t0 = time.time()
sim.init_sim()

# Meep's own eigenmode solver on this grid: isolated A (before B starts) and the
# two quasi-TE supermodes of the pair, for a like-for-like kappa.
def eig_n(x, band):
    em = sim.get_eigenmode(F0, mp.X, mp.Volume(center=mp.Vector3(x, 0, 0),
                           size=mp.Vector3(0, 2 * ymax, 2 * zmax)),
                           band_num=band, kpoint=mp.Vector3(F0 * 2.7), parity=mp.EVEN_Z)
    return em.k.x / F0

n_iso = eig_n(x0 - 1.0, 1)
n_plus, n_minus = eig_n(x0 + L / 2, 1), eig_n(x0 + L / 2, 2)

sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-5, maximum_run_time=900))
runtime = time.time() - t0

PA = np.array([mp.get_fluxes(f)[0] for f in flux_a])
PB = np.array([mp.get_fluxes(f)[0] for f in flux_b])

if mp.am_master():
    ex, ey, ez = (sim.get_dft_array(top, comp, 0) for comp in (mp.Ex, mp.Ey, mp.Ez))
    I_top = (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)   # (meep x, meep y)
    xs_cuts = []
    for d in xsec:
        a = [sim.get_dft_array(d, comp, 0) for comp in (mp.Ex, mp.Ey, mp.Ez)]
        xs_cuts.append(sum(np.abs(v) ** 2 for v in a))             # (meep y, meep z)
    np.savez_compressed(
        OUT / f"fdtd3d_res{RES}.npz", z=zs, PA=PA, PB=PB,
        top_I=I_top.T,                                   # rows = notes' x, cols = notes' z
        top_z=np.linspace(-1.0, L + 1.0, I_top.shape[0]),
        top_x=np.linspace(-ymax, ymax, I_top.shape[1]),
        xsec_I=np.array([v.T for v in xs_cuts]),          # rows = notes' y, cols = notes' x
        xsec_z=np.array(xsec_z),
        xsec_x=np.linspace(-ymax, ymax, xs_cuts[0].shape[0]),
        xsec_y=np.linspace(-zmax, zmax, xs_cuts[0].shape[1]),
    )
    tot = PA + PB
    json.dump({
        "resolution_px_per_um": RES, "runtime_s": runtime,
        "cell_um": [sx, sy, sz], "coupled_length_um": L,
        "n_iso_mpb_same_grid": n_iso, "n_plus_mpb": n_plus, "n_minus_mpb": n_minus,
        "kappa_mpb_same_grid": float(np.pi * (n_plus - n_minus) / LAM),
        "total_power_spread": float(tot[zs >= 0].std() / tot[zs >= 0].mean()),
    }, open(OUT / f"fdtd3d_res{RES}.json", "w"), indent=2)
    print(f"done in {runtime:.0f} s; n_iso {n_iso:.4f}, n+ {n_plus:.4f}, n- {n_minus:.4f}")
