"""Quasi-TE and quasi-TM modes of a silicon strip, with all six field components.

The 2-D modes come from femwell (full-vector finite elements). Its Nedelec
elements keep the tangential E continuous across an interface and let the normal
E jump, which is exactly the physics this comparison is about, so the field
discontinuities in the plots are solved for, not drawn in.

The 1-D slab TE and TM solutions are closed form and serve as the check.

Convention: fields are E(x, y) exp(j(wt - beta z)). Lengths in micrometres.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import indices as ix  # noqa: E402  (Sellmeier fits and the TE slab solver)


# --------------------------------------------------------------------------
# 1-D slab, both polarizations
#
#   TE (E along the interfaces, y):   h d - atan(gamma/h)                = m pi/2
#   TM (H along the interfaces, E across): h d - atan((n1/n2)^2 gamma/h) = m pi/2
#
# The only difference is the (n1/n2)^2 factor, which comes from matching
# (1/n^2) dH/dx instead of dE/dx at the boundary. That single factor is the
# whole TE/TM story in its simplest form.
# --------------------------------------------------------------------------


def slab_neff(lam, t, n1, n2, pol="TE", m=0):
    k0 = 2 * math.pi / lam
    d = t / 2
    r = 1.0 if pol == "TE" else (n1 / n2) ** 2

    def f(ne):
        h = k0 * math.sqrt(max(n1**2 - ne**2, 1e-15))
        g = k0 * math.sqrt(max(ne**2 - n2**2, 1e-15))
        return h * d - math.atan2(r * g, h) - m * math.pi / 2

    lo, hi = n2 + 1e-9, n1 - 1e-9
    if f(lo) * f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-13)


def slab_fields(lam, t, n1, n2, ne, x, pol="TE"):
    """Transverse E field across the slab for the fundamental mode.

    TE: the field is E_parallel, continuous everywhere.
    TM: H_parallel is continuous and E_normal = beta H / (w eps), so E jumps by
        (n1/n2)^2 at each face. Returned normalised to 1 at the centre.
    """
    k0 = 2 * math.pi / lam
    d = t / 2
    h = k0 * math.sqrt(n1**2 - ne**2)
    g = k0 * math.sqrt(ne**2 - n2**2)
    inside = np.abs(x) <= d
    shape = np.where(inside, np.cos(h * x), math.cos(h * d) * np.exp(-g * (np.abs(x) - d)))
    if pol == "TE":
        return shape
    n2map = np.where(inside, n1**2, n2**2)
    e = shape / n2map
    return e / e[np.argmin(np.abs(x))]


# --------------------------------------------------------------------------
# 2-D strip with femwell
# --------------------------------------------------------------------------


@dataclass
class Mode:
    label: str
    n_eff: float
    te_fraction: float
    dneff_dnsi: float           # sensitivity of n_eff to the silicon index
    obj: object                 # femwell mode
    grids: dict = field(default_factory=dict)


def solve(lam, w, h, num_modes=4, res=0.012, pad=1.4, n_core=None, n_clad=None):
    import shapely
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio

    n_core = float(ix.n_silicon(lam)) if n_core is None else n_core
    n_clad = float(ix.n_silica(lam)) if n_clad is None else n_clad
    polys = OrderedDict(
        core=shapely.box(-w / 2, -h / 2, w / 2, h / 2),
        clad=shapely.box(-w / 2 - pad, -h / 2 - pad, w / 2 + pad, h / 2 + pad),
    )
    mesh = from_meshio(mesh_from_OrderedDict(
        polys, {"core": {"resolution": res, "distance": 0.5}},
        default_resolution_max=0.15))
    b0 = Basis(mesh, ElementTriP0())
    eps = b0.zeros()
    for s, n in {"core": n_core, "clad": n_clad}.items():
        eps[b0.get_dofs(elements=s)] = n**2
    out = []
    for m in compute_modes(b0, eps, wavelength=lam, num_modes=num_modes, order=2):
        try:
            s = float(np.real(m.calculate_confinement_factor("core")))
        except Exception:
            s = float("nan")
        out.append(Mode("", float(np.real(m.n_eff)), float(np.real(m.te_fraction)), s, m))
    return out


def pick_te_tm(modes):
    """Fundamental quasi-TE = highest n_eff with mostly E_x; quasi-TM = mostly E_y."""
    te = max((m for m in modes if m.te_fraction > 0.5), key=lambda m: m.n_eff)
    tm = max((m for m in modes if m.te_fraction <= 0.5), key=lambda m: m.n_eff)
    te.label, tm.label = "quasi-TE0", "quasi-TM0"
    return te, tm


_PROBE_CACHE = {}


def _probes(basis, pts, tag):
    """Point-location is the slow part of probing; every mode from one solve shares
    the mesh, so the probe matrix is computed once per (mesh, element, grid)."""
    key = (id(basis.mesh), type(basis.elem).__name__, tag, pts.shape)
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = basis.probes(pts)
    return _PROBE_CACHE[key]


def sample(mode, xs, ys):
    """All six components of E and H on a regular (y, x) grid.

    Returns a dict of complex arrays Ex, Ey, Ez, Hx, Hy, Hz. The overall phase
    and sign of an eigenmode are arbitrary; they are fixed here so the dominant
    transverse E component is real and positive at the centre, which makes Ez and
    Hz come out purely imaginary (90 degrees behind), as they physically are.
    """
    m = mode.obj
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()])
    N = pts.shape[1]
    (Et, bt), (Ez, bz) = m.basis.split(m.E)
    (Ht, bht), (Hz, bhz) = m.basis.split(m.H)
    tag = (round(xs[0], 6), round(xs[-1], 6), round(ys[0], 6), round(ys[-1], 6))
    et = _probes(bt, pts, tag) @ Et
    ez = _probes(bz, pts, tag) @ Ez
    ht = _probes(bht, pts, tag) @ Ht
    hz = _probes(bhz, pts, tag) @ Hz
    g = {
        "Ex": et[:N], "Ey": et[N:], "Ez": ez,
        "Hx": ht[:N], "Hy": ht[N:], "Hz": hz,
    }
    g = {k: v.reshape(X.shape) for k, v in g.items()}
    cy, cx = np.argmin(np.abs(ys)), np.argmin(np.abs(xs))
    dom = "Ex" if mode.te_fraction > 0.5 else "Ey"
    ref = g[dom][cy, cx]
    phase = ref / abs(ref)
    escale = np.abs(g[dom]).max()
    hdom = "Hy" if dom == "Ex" else "Hx"
    hscale = np.abs(g[hdom]).max()
    for k in g:
        g[k] = g[k] / phase / (escale if k[0] == "E" else hscale)
    g["X"], g["Y"] = X, Y
    return g


def probe_line(mode, pts):
    """E components at arbitrary points (2, N), phase-fixed like sample()."""
    m = mode.obj
    N = pts.shape[1]
    (Et, bt), (Ez, bz) = m.basis.split(m.E)
    et = bt.probes(pts) @ Et
    return et[:N], et[N:], bz.probes(pts) @ Ez


def face_jump(mode, axis, face, n_in, n_out, steps=(0.002, 0.004, 0.006, 0.008)):
    """Normal-E jump across a face, by extrapolating each side linearly to the face.

    axis 0 = sidewall at x = face (normal component Ex),
    axis 1 = top face at y = face (normal component Ey).
    Returns (E_inside_at_face, E_outside_at_face, ratio, expected (n_in/n_out)^2).
    """
    s = np.array(steps)
    if axis == 0:
        pin = np.vstack([face - s, np.zeros_like(s)])
        pout = np.vstack([face + s, np.zeros_like(s)])
        comp = 0
    else:
        pin = np.vstack([np.zeros_like(s), face - s])
        pout = np.vstack([np.zeros_like(s), face + s])
        comp = 1
    ein = np.real(probe_line(mode, pin)[comp])
    eout = np.real(probe_line(mode, pout)[comp])
    a = np.polyfit(s, ein, 1)[1]
    b = np.polyfit(s, eout, 1)[1]
    return a, b, b / a, (n_in / n_out) ** 2


def group_index(lam, w, h, pick, dlam=0.02, **kw):
    """n_g = n_eff - lam dn_eff/dlam with Sellmeier materials moving too."""
    vals = []
    for l in (lam - dlam, lam + dlam):
        te, tm = pick_te_tm(solve(l, w, h, **kw))
        vals.append((te if pick == "TE" else tm).n_eff)
    te, tm = pick_te_tm(solve(lam, w, h, **kw))
    n0 = (te if pick == "TE" else tm).n_eff
    return n0, n0 - lam * (vals[1] - vals[0]) / (2 * dlam)
