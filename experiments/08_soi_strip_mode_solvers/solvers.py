"""Thin, convention-unifying wrappers around the two full-vector mode solvers.

Convention (matches docs/NOTES.md): x = width direction, y = height direction,
z = propagation direction.  A mode is  E(x,y) e^{-j beta z},  n_eff = beta/k0.

  * Tidy3D's local FDFD mode solver (finite differences on a Yee grid).  Tidy3D
    propagates along ITS x-axis, so its (y, z) are our (x, y) and its Ex is our
    longitudinal E_z.  The wrapper relabels everything.
  * femwell (FEM, Nedelec/Lagrange elements on a gmsh triangle mesh conforming to
    the core boundary).  femwell already uses x = width, y = height, z = propagation.

Both wrappers return a list of `Mode` records (guided modes only, sorted by
descending n_eff) carrying the same fields so run.py can compare them 1:1.

Confinement, three different "Gammas" (they are NOT the same number):
  gamma_P : fraction of the time-averaged Poynting flux S_z inside the silicon core
            (the textbook's Gamma = 0.85 is this one).
  gamma_E : fraction of the electric energy eps|E|^2 inside the core.
  S_core  : dn_eff/dn_core, the true first-order sensitivity of n_eff to the core
            index.  Perturbation theory (notes 25 energy density + 22 group index)
            gives S_core = gamma_E * n_g,wg / n_core, and femwell's
            `calculate_confinement_factor` is exactly this quantity.  It is what
            the thermo-optic shift needs.
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

# common output grid (um) shared by both solvers for plots and cuts
XS = np.round(np.arange(-1.0, 1.0001, 0.01), 4)
YS = np.round(np.arange(-0.6, 0.6001, 0.01), 4)


@dataclass
class Mode:
    solver: str
    label: str            # TE0, TM0, TE1, ...
    n_eff: float
    te_fraction: float
    gamma_P: float        # Poynting power fraction in core
    gamma_E: float        # electric-energy fraction in core
    gamma_E_clad: float
    n_g_wg: float | None = None   # group index at fixed (non-dispersive) materials
    S_core: float | None = None   # dn_eff/dn_core (femwell: exact functional; tidy3d: gamma_E n_g/n_core)
    S_clad: float | None = None
    fields: dict = field(default_factory=dict)  # 'Ex','Ey','Ez','Sz' on (XS, YS) grid


def _classify(records, n_clad):
    """Keep guided modes (n_eff > n_clad) and name them TE0, TM0, TE1 ... in order of n_eff."""
    out, counts = [], {"TE": 0, "TM": 0}
    for r in sorted(records, key=lambda r: -r.n_eff):
        if r.n_eff <= n_clad + 1e-6:
            continue
        pol = "TE" if r.te_fraction >= 0.5 else "TM"
        r.label = f"{pol}{counts[pol]}"
        counts[pol] += 1
        out.append(r)
    return out


def _fix_phase(Ex, Ey, Ez):
    """Rotate the global phase so the dominant transverse component is real and positive at its peak."""
    Et = np.stack([Ex, Ey])
    idx = np.unravel_index(np.argmax(np.abs(Et)), Et.shape)
    ph = np.exp(-1j * np.angle(Et[idx]))
    return Ex * ph, Ey * ph, Ez * ph


# --------------------------------------------------------------------------- Tidy3D
def solve_tidy3d(width, height, lam, n_core, n_clad, num_modes=3, dl=0.01,
                 box=(2.61, 2.01), want_fields=False, group_index=True):
    """Tidy3D local FDFD mode solve on a uniform Yee grid of pitch `dl`.

    Grid alignment matters: the local solver does not do subpixel averaging, so a material
    edge that falls exactly ON a grid line is assigned to one side or the other by a
    rounding rule that flips between geometries (+-0.5 cell of effective width, i.e.
    +-0.005 in n_eff for this strip).  With `box` an ODD multiple of dl the grid lines sit
    at (k + 1/2) dl and every edge at a multiple of dl lands mid-cell, where the sampling
    is unambiguous.  Checked: 2.61 x 2.01 at dl = 0.01 gives the same n_eff as dl = 0.005
    (+0.0014 above the converged FEM value at every width)."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    td.config.logging.level = "ERROR"

    core = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(td.inf, width, height)),
                        medium=td.Medium(permittivity=n_core**2))
    sim = td.Simulation(size=(2.0, box[0], box[1]), grid_spec=td.GridSpec.uniform(dl=dl),
                        structures=[core], medium=td.Medium(permittivity=n_clad**2),
                        run_time=1e-12,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, 0), size=(0, box[0], box[1])),
                    mode_spec=td.ModeSpec(num_modes=num_modes, group_index_step=group_index),
                    freqs=[td.C_0 / lam])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = ms.solve()

    yc = d.Ex.coords["y"].values          # our x (width)
    zc = d.Ex.coords["z"].values          # our y (height)
    Yg, Zg = np.meshgrid(yc, zc, indexing="ij")
    in_core = (np.abs(Yg) <= width / 2 + 1e-9) & (np.abs(Zg) <= height / 2 + 1e-9)
    eps = np.where(in_core, n_core**2, n_clad**2)
    # trapezoid weights: grid points that sit exactly on the core boundary count half
    # (quarter at the corners), otherwise a 10 nm grid over-counts the core by ~2 %.
    wy = np.where(np.isclose(np.abs(Yg), width / 2, atol=1e-9), 0.5, 1.0)
    wz = np.where(np.isclose(np.abs(Zg), height / 2, atol=1e-9), 0.5, 1.0)
    core_w = in_core * wy * wz

    recs = []
    for i in range(num_modes):
        sel = dict(f=0, mode_index=i)
        comp = {c: getattr(d, c).isel(**sel).values.squeeze() for c in ("Ex", "Ey", "Ez")}
        E2 = sum(np.abs(v) ** 2 for v in comp.values())
        Sz = np.real(d.complex_poynting.isel(**sel).values.squeeze())
        gP = (Sz * core_w).sum() / Sz.sum()
        w = eps * E2
        gE = (w * core_w).sum() / w.sum()
        ng = float(d.n_group.values[0, i]) if group_index else None
        r = Mode(solver="tidy3d", label="", n_eff=float(np.real(d.n_eff.values[0, i])),
                 te_fraction=float(d.pol_fraction.te.values[0, i]),
                 gamma_P=float(gP), gamma_E=float(gE), gamma_E_clad=float(1 - gE), n_g_wg=ng)
        if ng is not None:
            r.S_core = gE * ng / n_core
            r.S_clad = (1 - gE) * ng / n_clad
        if want_fields:
            # relabel: tidy3d (Ey, Ez, Ex) -> ours (Ex, Ey, Ez); sample on the common grid
            ex = d.Ey.isel(**sel).squeeze(drop=True).interp(y=XS, z=YS).values
            ey = d.Ez.isel(**sel).squeeze(drop=True).interp(y=XS, z=YS).values
            ez = d.Ex.isel(**sel).squeeze(drop=True).interp(y=XS, z=YS).values
            sz = d.complex_poynting.isel(**sel).squeeze(drop=True).interp(y=XS, z=YS).values
            ex, ey, ez = _fix_phase(ex, ey, ez)
            r.fields = dict(Ex=ex, Ey=ey, Ez=ez, Sz=np.real(sz))
        recs.append(r)
    return _classify(recs, n_clad)


# --------------------------------------------------------------------------- femwell
@dataclass
class FemMesh:
    basis0: object
    core_dofs: object
    clad_dofs: object
    nelements: int
    probes: dict = field(default_factory=dict)   # cached point-evaluation matrices


def femwell_mesh(width, height, resolution=0.02, box=1.5, distance=0.4, resolution_max=0.25):
    import shapely
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio
    from femwell.mesh import mesh_from_OrderedDict

    polys = OrderedDict(core=shapely.box(-width / 2, -height / 2, width / 2, height / 2),
                        clad=shapely.box(-box, -box, box, box))
    mesh = from_meshio(mesh_from_OrderedDict(
        polys, {"core": {"resolution": resolution, "distance": distance}},
        default_resolution_max=resolution_max))
    basis0 = Basis(mesh, ElementTriP0())
    return FemMesh(basis0, basis0.get_dofs(elements="core"), basis0.get_dofs(elements="clad"),
                   mesh.nelements)


def solve_femwell(fm: FemMesh, lam, n_core, n_clad, num_modes=3, want_fields=False, order=2, cut_x=None):
    from femwell.maxwell.waveguide import compute_modes
    from skfem import ElementDG, ElementTriP1, ElementVector, Functional

    eps = fm.basis0.zeros()
    eps[fm.core_dofs] = n_core**2
    eps[fm.clad_dofs] = n_clad**2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modes = compute_modes(fm.basis0, eps, wavelength=lam, num_modes=num_modes, order=order)

    @Functional
    def energy(w):
        return w["epsilon"] * (np.abs(w["E"][0][0]) ** 2 + np.abs(w["E"][0][1]) ** 2
                               + np.abs(w["E"][1]) ** 2)

    recs = []
    for m in modes:
        def en(elements=None):
            b = m.basis if elements is None else m.basis.with_elements(elements)
            be = m.basis_epsilon_r if elements is None else m.basis_epsilon_r.with_elements(elements)
            return float(np.real(energy.assemble(b, E=b.interpolate(m.E),
                                                 epsilon=be.interpolate(m.epsilon_r))))
        etot = en()
        gE = en("core") / etot
        r = Mode(solver="femwell", label="", n_eff=float(np.real(m.n_eff)),
                 te_fraction=float(m.te_fraction),
                 gamma_P=float(np.real(m.calculate_power(elements="core"))),
                 gamma_E=gE, gamma_E_clad=en("clad") / etot,
                 S_core=float(np.real(m.calculate_confinement_factor("core"))),
                 S_clad=float(np.real(m.calculate_confinement_factor("clad"))))
        if want_fields:
            def gridded(vec):
                """Evaluate a femwell (E or H) DOF vector on the common (XS, YS) grid.

                Transverse part: project onto a discontinuous P1 vector basis (as femwell's own
                plot_component does), longitudinal part: already a Lagrange P2 scalar.  Point
                evaluation uses skfem probe matrices, built once per mesh and cached."""
                (vt, bt), (vz, bz) = m.basis.split(vec)
                pb = bt.with_element(ElementVector(ElementDG(ElementTriP1())))
                if "t" not in fm.probes:
                    pts = np.vstack([g.ravel() for g in np.meshgrid(XS, YS, indexing="ij")])
                    bx0, _ = pb.split_bases()[0], None
                    fm.probes["t"] = bx0.probes(pts)
                    fm.probes["z"] = bz.probes(pts)
                Pt, Pz = fm.probes["t"], fm.probes["z"]
                comps = []
                for part in (np.real, np.imag):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vxy = pb.project(part(bt.interpolate(vt)))
                    (vx, _), (vy, _) = pb.split(vxy)
                    comps.append([Pt @ vx, Pt @ vy, Pz @ part(vz)])
                re, im = comps
                return [(a + 1j * b).reshape(len(XS), len(YS)) for a, b in zip(re, im)]

            ex, ey, ez = gridded(m.E)
            hx, hy, hz = gridded(m.H)
            sz = 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))
            ex, ey, ez = _fix_phase(ex, ey, ez)
            r.fields = dict(Ex=ex, Ey=ey, Ez=ez, Sz=sz)
            if cut_x is not None:
                # E_x along y = 0 at arbitrary x (used to probe the sidewall discontinuity to 0.1 nm)
                (vt, bt), _ = m.basis.split(m.E)
                pb = bt.with_element(ElementVector(ElementDG(ElementTriP1())))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vxy = pb.project(np.real(bt.interpolate(vt)))
                (vx, bx), _ = pb.split(vxy)
                cut = bx.probes(np.vstack([cut_x, np.zeros_like(cut_x)])) @ vx
                cut = cut * np.sign(cut[np.argmax(np.abs(cut))])
                r.fields["Ex_cut"] = (np.asarray(cut_x), cut)
        recs.append(r)
    return _classify(recs, n_clad)


def group_index_fd(solve_at_lambda, lam, dlam=0.005):
    """n_g = n_eff - lam dn_eff/dlam by central difference; solve_at_lambda(lam) -> n_eff."""
    n0 = solve_at_lambda(lam)
    dn = (solve_at_lambda(lam + dlam) - solve_at_lambda(lam - dlam)) / (2 * dlam)
    return n0 - lam * dn, n0


def find_mode(modes, label):
    for m in modes:
        if m.label == label:
            return m
    return None
