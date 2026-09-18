"""Analytic TE slab and coupled-slab (directional coupler) helpers, pure numpy/scipy.

Used by run.py (Meep interpreter) and by sax_part.py / gds_part.py (.venv interpreter),
so it must import nothing beyond numpy and scipy.

Conventions follow docs/NOTES.md: e^{jωt}, +z wave e^{-jβz}, x perpendicular to the
slab boundaries.  Lengths in µm, propagation constants in rad/µm.

Implements
  * notes §16/§19: single symmetric slab, even TE eigenvalue  h tan(h d) = γ
  * notes §20/§27: supermodes of two identical slabs by an exact transfer matrix
    (5-layer TE eigenproblem), giving β+ (even) and β- (odd) and κ_c = (β+ - β-)/2
  * notes §27: closed-form coupled-mode coefficient for two slabs (Yariv/Marcuse)
      κ_c = 2 h² γ e^{-γ s} / [ β (W + 2/γ) (h² + γ²) ]      (W = full slab width, s = gap)
  * point-coupler effective length of a ring against a straight bus (capstone)
      gap(z) ≈ g0 + z²/(2R)  ->  ∫ e^{-γ z²/(2R)} dz = sqrt(2πR/γ) = L_eff
"""
import numpy as np
from scipy.optimize import brentq


def k0_per_um(lam_um):
    return 2 * np.pi / lam_um


# ---------------------------------------------------------------- single slab (notes §16-§19)
def single_slab_te(n1=3.50, n2=1.45, width=0.22, lam=1.31):
    """Fundamental even TE mode of a symmetric slab: returns dict(n_eff, beta, h, gamma).

    Solves h tan(h d) = γ with h² + γ² = k0²(n1² - n2²) (the circle of notes §19), d = width/2.
    """
    k0 = k0_per_um(lam)
    d = width / 2
    V = k0 * d * np.sqrt(n1**2 - n2**2)

    def f(hd):
        gd = np.sqrt(max(V**2 - hd**2, 0.0))
        return hd * np.tan(hd) - gd

    hd = brentq(f, 1e-9, min(V, np.pi / 2) - 1e-9)   # fundamental branch 0 < hd < π/2
    h = hd / d
    gamma = np.sqrt(V**2 - hd**2) / d
    beta = np.sqrt((n1 * k0) ** 2 - h**2)
    return dict(n_eff=beta / k0, beta=beta, h=h, gamma=gamma, decay_len=1 / gamma, V=V)


# ---------------------------------------------------------------- multilayer TE transfer matrix
def _te_layer_matrix(n, t, beta, k0):
    """2x2 matrix propagating (F, F') across a layer of index n and thickness t (notes §18:
    F and F' are both continuous, so the state vector is (F, F'))."""
    kx2 = (n * k0) ** 2 - beta**2
    if kx2 > 0:
        k = np.sqrt(kx2)
        return np.array([[np.cos(k * t), np.sin(k * t) / k],
                         [-k * np.sin(k * t), np.cos(k * t)]])
    g = np.sqrt(-kx2)
    return np.array([[np.cosh(g * t), np.sinh(g * t) / g],
                     [g * np.sinh(g * t), np.cosh(g * t)]])


def _te_dispersion(beta, layers, n_clad, k0):
    """Zero when the field in the top cladding is purely decaying (notes §16: bounded solution).

    Start in the bottom cladding with the bounded solution F = e^{+γ x}: (F, F') = (1, γ).
    Propagate through the finite layers.  In the top cladding F = A e^{-γx} + B e^{+γx};
    B ∝ F' + γ F must vanish.
    """
    gamma = np.sqrt(beta**2 - (n_clad * k0) ** 2)
    v = np.array([1.0, gamma])
    for n, t in layers:
        v = _te_layer_matrix(n, t, beta, k0) @ v
    return v[1] + gamma * v[0]


def multilayer_te_modes(layers, n_clad=1.45, lam=1.31, n_scan=4000):
    """All bound TE modes of a stack of (index, thickness) layers in a uniform cladding.

    Returns n_eff values sorted descending (band 1 first).
    """
    k0 = k0_per_um(lam)
    n_max = max(n for n, _ in layers)
    ne = np.linspace(n_clad + 1e-4, n_max - 1e-4, n_scan)
    vals = np.array([_te_dispersion(n * k0, layers, n_clad, k0) for n in ne])
    roots = []
    for i in range(len(ne) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]) and np.isfinite(vals[i]) and np.isfinite(vals[i + 1]):
            r = brentq(lambda n: _te_dispersion(n * k0, layers, n_clad, k0), ne[i], ne[i + 1], xtol=1e-13)
            roots.append(r)
    return np.array(sorted(roots, reverse=True))


def te_profile(n_eff, layers, n_clad, lam, x, x_start):
    """Transverse profile F(x) of a TE mode of the stack (layers start at x_start), normalised to max |F| = 1."""
    k0 = k0_per_um(lam)
    beta = n_eff * k0
    gamma = np.sqrt(beta**2 - (n_clad * k0) ** 2)
    # boundaries
    edges = [x_start]
    for _, t in layers:
        edges.append(edges[-1] + t)
    # state at each boundary
    states = [np.array([1.0, gamma])]
    for (n, t) in layers:
        states.append(_te_layer_matrix(n, t, beta, k0) @ states[-1])
    F = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < edges[0]:
            F[i] = np.exp(gamma * (xi - edges[0]))
        elif xi >= edges[-1]:
            F[i] = states[-1][0] * np.exp(-gamma * (xi - edges[-1]))
        else:
            j = np.searchsorted(edges, xi, side="right") - 1
            n, _ = layers[j]
            F[i] = (_te_layer_matrix(n, xi - edges[j], beta, k0) @ states[j])[0]
    return F / np.max(np.abs(F))


def coupled_slabs_layers(gap, width=0.22, n1=3.50):
    return [(n1, width), (1.45, gap), (n1, width)]


def supermodes(gap, n1=3.50, n2=1.45, width=0.22, lam=1.31, width2=None):
    """Even/odd supermode indices of two slabs (notes §27) and κ_c = (β+ - β-)/2.

    width2 (default = width) makes the second slab different: the supermodes are then no longer
    symmetric/antisymmetric and (β+ - β-)/2 = sqrt(κ² + Δβ²/4) with Δβ = β1 - β2 of the isolated
    slabs; the returned 'kappa' is that generalised rate and 'P2_max' = κ²/(κ² + Δβ²/4) the maximum
    fraction transferred (1 for identical slabs).
    """
    width2 = width if width2 is None else width2
    layers = [(n1, width), (n2, gap), (n1, width2)]
    ne = multilayer_te_modes(layers, n_clad=n2, lam=lam)
    k0 = k0_per_um(lam)
    n_even, n_odd = ne[0], ne[1]
    kappa = k0 * (n_even - n_odd) / 2
    dbeta = single_slab_te(n1, n2, width, lam)["beta"] - single_slab_te(n1, n2, width2, lam)["beta"]
    kappa_pure2 = max(kappa**2 - dbeta**2 / 4, 0.0)
    return dict(n_even=n_even, n_odd=n_odd, beta_even=n_even * k0, beta_odd=n_odd * k0,
                kappa=kappa, L_c=np.pi / (2 * kappa), layers=layers, dbeta=dbeta,
                P2_max=kappa_pure2 / kappa**2 if kappa > 0 else 1.0)


# ---------------------------------------------------------------- coupled-mode closed form (notes §27)
def cmt_kappa(gap, n1=3.50, n2=1.45, width=0.22, lam=1.31):
    """Coupled-mode-theory κ_c for two identical TE slabs separated by `gap` (Yariv, Optical Electronics)."""
    m = single_slab_te(n1, n2, width, lam)
    h, g, b = m["h"], m["gamma"], m["beta"]
    return 2 * h**2 * g * np.exp(-g * gap) / (b * (width + 2 / g) * (h**2 + g**2))


# ---------------------------------------------------------------- point coupler (capstone)
def point_coupler_length(R, gamma):
    """L_eff = sqrt(2πR/γ): the length of a parallel coupler with the same ∫κ dz as a ring of
    radius R touching a straight bus, when κ(gap) ∝ e^{-γ gap} and gap(z) ≈ g0 + z²/(2R)."""
    return np.sqrt(2 * np.pi * R / gamma)
