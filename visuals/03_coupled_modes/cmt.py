"""Coupled-mode theory for two silicon strips, built from real mode solves.

Two models are solved on one shared mesh, so every overlap integral is exact
(no interpolation between meshes):

  scalar  the model the notes derive with:  lap_t u + (k0^2 n^2 - beta^2) u = 0,
          solved with P2 finite elements (scikit-fem). This is the world in which
          the formula  kappa = k0^2/(2 beta) * int_B Delta_B u_B u_A / int u_B^2  holds.
  vector  full Maxwell (femwell, Nedelec + P2 elements). This is the real silicon
          strip; its coupling formula is the same idea with E-vectors and power
          normalisation.

In both models the exact answer is available independently: solve the pair of
guides as one structure, take its two supermodes, kappa = (beta_+ - beta_-)/2.

Coordinates: x across the chip (A at negative x, B at positive x), y vertical,
z along the guides. Convention exp(j(wt - beta z)), as in the notes. Lengths in um.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import indices as ix  # noqa: E402  (Sellmeier fits)

LAM = 1.31          # um
W, H = 0.50, 0.22   # strip width and height, um
GAP = 0.15          # the coupler used throughout, um
K0 = 2 * math.pi / LAM
N_SI = float(ix.n_silicon(LAM))
N_OX = float(ix.n_silica(LAM))
DN2 = N_SI**2 - N_OX**2          # the height of the Delta masks
C0, EPS0 = 299792458.0, 8.8541878128e-12


def centres(gap):
    c = (W + gap) / 2
    return -c, c


# --------------------------------------------------------------------------
# mesh shared by every solve at one gap
# --------------------------------------------------------------------------


@dataclass
class Pair:
    gap: float
    mesh: object
    b0: object            # P0 basis for femwell (permittivity)
    b2: object            # P2 basis for the scalar model
    s0: object            # P0 basis with b2's quadrature
    xa: float
    xb: float
    cache: dict = field(default_factory=dict)


def build(gap=GAP, res=0.015, pad_x=1.3, pad_y=1.1):
    import shapely
    from femwell.mesh import mesh_from_OrderedDict
    from skfem import Basis, ElementTriP0, ElementTriP2
    from skfem.io.meshio import from_meshio

    xa, xb = centres(gap)
    polys = OrderedDict(
        coreA=shapely.box(xa - W / 2, -H / 2, xa + W / 2, H / 2),
        coreB=shapely.box(xb - W / 2, -H / 2, xb + W / 2, H / 2),
        clad=shapely.box(xa - W / 2 - pad_x, -H / 2 - pad_y, xb + W / 2 + pad_x, H / 2 + pad_y),
    )
    spec = {"resolution": res, "distance": 0.6}
    mesh = from_meshio(mesh_from_OrderedDict(
        polys, {"coreA": spec, "coreB": spec}, default_resolution_max=0.15))
    b2 = Basis(mesh, ElementTriP2())
    return Pair(gap, mesh, Basis(mesh, ElementTriP0()), b2, b2.with_element(ElementTriP0()), xa, xb)


def eps_map(basis0, a=True, b=True, delta_only=None):
    """n^2 on P0 dofs: cladding everywhere, silicon in the cores that exist.

    delta_only='A' or 'B' returns just that Delta mask (n_Si^2 - n_ox^2 inside the
    core, 0 elsewhere): the "mask" of the notes.
    """
    if delta_only:
        e = basis0.zeros()
        e[basis0.get_dofs(elements="core" + delta_only)] = DN2
        return e
    e = basis0.zeros() + N_OX**2
    if a:
        e[basis0.get_dofs(elements="coreA")] = N_SI**2
    if b:
        e[basis0.get_dofs(elements="coreB")] = N_SI**2
    return e


# --------------------------------------------------------------------------
# scalar model (the notes' equation)
# --------------------------------------------------------------------------


def _scalar_mats(p):
    if "scalar_mats" not in p.cache:
        from skfem import BilinearForm, asm
        from skfem.helpers import dot, grad

        @BilinearForm
        def lap(u, v, w):
            return dot(grad(u), grad(v))

        @BilinearForm
        def mass(u, v, w):
            return u * v

        @BilinearForm
        def wmass(u, v, w):
            return w.e * u * v

        p.cache["scalar_mats"] = (asm(lap, p.b2), asm(mass, p.b2),
                                  lambda e: asm(wmass, p.b2, e=p.s0.interpolate(e)))
    return p.cache["scalar_mats"]


def scalar_modes(p, a=True, b=True, k=2):
    """Largest-beta scalar modes. Returns (n_eff array, vectors) with each vector
    normalised to int u^2 dA = 1 and its sign fixed (positive where it is largest
    near the core centres)."""
    K, M, N = _scalar_mats(p)
    A = K0**2 * N(eps_map(p.s0, a, b)) - K          # A u = beta^2 M u
    vals, vecs = sla.eigsh(A, k=k, M=M, sigma=(K0 * N_SI) ** 2, which="LM")
    o = np.argsort(-vals)
    vals, vecs = vals[o], vecs[:, o]
    for i in range(vecs.shape[1]):
        v = vecs[:, i] / math.sqrt(vecs[:, i] @ M @ vecs[:, i])
        # sign: positive at the centre of the core that exists (A if both)
        x0 = p.xa if a else p.xb
        ref = (p.b2.probes(np.array([[x0], [0.0]])) @ v)[0]
        vecs[:, i] = v * (1 if ref >= 0 else -1)
    return np.sqrt(vals) / K0, vecs


def scalar_kappa_overlap(p, uA, uB, nA):
    """The notes' formula, evaluated in the scalar model."""
    K, M, N = _scalar_mats(p)
    beta = K0 * nA
    num = uB @ N(eps_map(p.s0, delta_only="B")) @ uA
    return K0**2 / (2 * beta) * num / (uB @ M @ uB)


def scalar_residual(p, uA, nA):
    """Plug isolated-A's scalar mode into the two-guide equation and return what
    is left over, as a nodal field: (k0^2 n_pair^2 - beta_A^2 + lap) u_A.

    In the weak form this is r = (k0^2 N_pair - K - beta_A^2 M) u_A, a load vector;
    solving M f = r turns it into the field f you can plot (an L2 projection; P2
    elements cannot be mass-lumped)."""
    K, M, N = _scalar_mats(p)
    r = (K0**2 * N(eps_map(p.s0, True, True)) - K - (K0 * nA) ** 2 * M) @ uA
    return sla.spsolve(M.tocsc(), r)


# --------------------------------------------------------------------------
# vector model (femwell)
# --------------------------------------------------------------------------


@dataclass
class VMode:
    label: str
    n_eff: float
    te_fraction: float
    obj: object

    @property
    def beta(self):
        return K0 * self.n_eff


def vector_modes(p, a=True, b=True, num_modes=4):
    from femwell.maxwell.waveguide import compute_modes

    out = []
    for m in compute_modes(p.b0, eps_map(p.b0, a, b), wavelength=LAM,
                           num_modes=num_modes, order=2):
        out.append(VMode("", float(np.real(m.n_eff)), float(np.real(m.te_fraction)), m))
    return out


def classify(modes):
    te = sorted([m for m in modes if m.te_fraction > 0.5], key=lambda m: -m.n_eff)
    tm = sorted([m for m in modes if m.te_fraction <= 0.5], key=lambda m: -m.n_eff)
    for i, m in enumerate(te):
        m.label = f"TE{i}"
    for i, m in enumerate(tm):
        m.label = f"TM{i}"
    return te, tm


def power(m):
    """femwell's calculate_power is Re int E x H* . z, i.e. twice the physical power."""
    return float(np.real(m.obj.calculate_power()))


def vector_kappa_overlap(p, mA, mB):
    """kappa_BA = (w eps0 / 4) int Delta_eps_B  E_B* . E_A dA  with each mode carrying
    1 W, written in femwell's units (power = Re int E x H*, lengths in um):
    kappa = (w eps0 / 2) * overlap / sqrt(P_A P_B), returned in rad/um."""
    ov = mB.obj.calculate_coupling_coefficient(mA.obj, eps_map(p.b0, delta_only="B"))
    w = 2 * math.pi * C0 / (LAM * 1e-6)
    return abs(ov) * w * EPS0 / 2 / math.sqrt(power(mA) * power(mB)) * 1e-6, ov


def vector_overlap_abs(p, mA, mB):
    """|int Delta_eps_B E_B* . E_A| / sqrt(P_A P_B): the bare projection, for comparing
    how well A's leftover field matches different modes of B."""
    ov = mB.obj.calculate_coupling_coefficient(mA.obj, eps_map(p.b0, delta_only="B"))
    return abs(ov) / math.sqrt(power(mA) * power(mB))


def power_overlap(m1, m2):
    """Power-normalised mode overlap <m1|m2> (femwell's E x H* form)."""
    return complex(m1.obj.calculate_overlap(m2.obj)) / math.sqrt(power(m1) * power(m2))


# --------------------------------------------------------------------------
# sampling on regular grids
# --------------------------------------------------------------------------


_PROBES = {}


def _probe(basis, pts, tag):
    key = (id(basis.mesh), type(basis.elem).__name__, tag, pts.shape)
    if key not in _PROBES:
        _PROBES[key] = basis.probes(pts)
    return _PROBES[key]


def grid(xs, ys):
    X, Y = np.meshgrid(xs, ys)
    return X, Y, np.vstack([X.ravel(), Y.ravel()]), (round(xs[0], 6), round(xs[-1], 6), len(xs),
                                                     round(ys[0], 6), round(ys[-1], 6), len(ys))


def sample_scalar(p, u, xs, ys):
    X, Y, pts, tag = grid(xs, ys)
    return (_probe(p.b2, pts, tag) @ u).reshape(X.shape)


def sample_vector(m, xs, ys, phase_ref=None):
    """Ex, Ey, Ez on a (y, x) grid. The overall phase of an eigenmode is arbitrary;
    it is fixed so that Ex is real and positive at phase_ref (x, y)."""
    X, Y, pts, tag = grid(xs, ys)
    N = pts.shape[1]
    (Et, bt), (Ez, bz) = m.obj.basis.split(m.obj.E)
    (Ht, bht), _ = m.obj.basis.split(m.obj.H)
    et = _probe(bt, pts, tag) @ Et
    ez = _probe(bz, pts, tag) @ Ez
    ht = _probe(bht, pts, tag) @ Ht
    g = {"Ex": et[:N].reshape(X.shape), "Ey": et[N:].reshape(X.shape), "Ez": ez.reshape(X.shape),
         "Hx": ht[:N].reshape(X.shape), "Hy": ht[N:].reshape(X.shape)}
    if phase_ref is not None:
        pr = np.array([[phase_ref[0]], [phase_ref[1]]])
        ref = (bt.probes(pr) @ Et)[0]
        ph = ref / abs(ref)
        g = {k: v / ph for k, v in g.items()}
    return g


def line_vector(m, xs, y=0.0, phase_ref=None):
    pts = np.vstack([xs, np.full_like(xs, y)])
    (Et, bt), _ = m.obj.basis.split(m.obj.E)
    ex = (bt.probes(pts) @ Et)[: len(xs)]
    if phase_ref is not None:
        ref = (bt.probes(np.array([[phase_ref[0]], [phase_ref[1]]])) @ Et)[0]
        ex = ex / (ref / abs(ref))
    return ex


def line_scalar(p, u, xs, y=0.0):
    pts = np.vstack([xs, np.full_like(xs, y)])
    return p.b2.probes(pts) @ u


# --------------------------------------------------------------------------
# one-call summary of a gap: exact and overlap kappa in both models
# --------------------------------------------------------------------------


def kappa_at_gap(gap, res=0.02):
    p = build(gap, res=res)
    nA, uA = scalar_modes(p, True, False, 1)
    nB, uB = scalar_modes(p, False, True, 1)
    ns, _ = scalar_modes(p, True, True, 2)
    vA = classify(vector_modes(p, True, False, 3))[0][0]
    vB = classify(vector_modes(p, False, True, 3))[0][0]
    te, _ = classify(vector_modes(p, True, True, 4))
    kv, _ = vector_kappa_overlap(p, vA, vB)
    return {
        "gap_um": gap,
        "scalar": {"n_iso": float(nA[0]), "n_plus": float(ns[0]), "n_minus": float(ns[1]),
                   "kappa_exact": math.pi * (ns[0] - ns[1]) / LAM,
                   "kappa_overlap": float(scalar_kappa_overlap(p, uA[:, 0], uB[:, 0], nA[0]))},
        "vector": {"n_iso": vA.n_eff, "n_plus": te[0].n_eff, "n_minus": te[1].n_eff,
                   "kappa_exact": math.pi * (te[0].n_eff - te[1].n_eff) / LAM,
                   "kappa_overlap": kv},
    }
