"""Analytic TE modes of the symmetric dielectric slab (docs/NOTES.md sections 16-23).

Core index n1 for |x| < d, cladding n2 outside, propagation along z, field E = y F(x) e^{-j beta z}.
Even modes satisfy  h tan(h d) = gamma,  odd modes  -h cot(h d) = gamma,
with  h^2 + gamma^2 = k0^2 (n1^2 - n2^2)  (the "circle" of radius V/d).
Pure numpy/scipy so it runs under both interpreters.
"""
import numpy as np
from scipy.optimize import brentq


def solve_te_modes(n1, n2, thickness_um, lambda_um):
    """Return a list of dicts (one per guided TE mode, fundamental first)."""
    d = thickness_um / 2
    k0 = 2 * np.pi / lambda_um
    V = k0 * d * np.sqrt(n1**2 - n2**2)      # normalised frequency
    modes = []
    # Roots live in u = h d in (m pi/2, (m+1) pi/2), m = 0,1,... while u < V
    m = 0
    while m * np.pi / 2 < V:
        lo = m * np.pi / 2 + 1e-9
        hi = min((m + 1) * np.pi / 2 - 1e-9, V - 1e-12)
        if hi <= lo:
            break
        if m % 2 == 0:
            f = lambda u: u * np.tan(u) - np.sqrt(V**2 - u**2)
        else:
            f = lambda u: -u / np.tan(u) - np.sqrt(V**2 - u**2)
        try:
            u = brentq(f, lo, hi)
        except ValueError:
            break
        h = u / d
        gamma = np.sqrt(V**2 - u**2) / d
        beta = np.sqrt(n1**2 * k0**2 - h**2)
        modes.append(dict(
            order=m, parity="even" if m % 2 == 0 else "odd",
            beta_per_um=beta, neff=beta / k0, h_per_um=h, gamma_per_um=gamma,
            decay_len_nm=1e3 / gamma, lambda_g_um=2 * np.pi / beta, V=V, d_um=d,
            edge_over_peak=np.cos(u) if m % 2 == 0 else np.sin(u),
        ))
        m += 1
    return modes


def profile(mode, x_um):
    """F(x): unit peak in the core (even) or unit amplitude sin (odd)."""
    d, h, g = mode["d_um"], mode["h_per_um"], mode["gamma_per_um"]
    x = np.asarray(x_um, dtype=float)
    F = np.empty_like(x)
    core = np.abs(x) <= d
    if mode["parity"] == "even":
        F[core] = np.cos(h * x[core])
        edge = np.cos(h * d)
        F[~core] = edge * np.exp(-g * (np.abs(x[~core]) - d))
    else:
        F[core] = np.sin(h * x[core])
        edge = np.sin(h * d)
        F[~core] = np.sign(x[~core]) * edge * np.exp(-g * (np.abs(x[~core]) - d))
    return F


def confinement(mode, n_pts=20001, span_factor=30):
    """Fraction of the longitudinal Poynting flux inside the core.

    For a TE slab mode S_z(x) = (beta / 2 omega mu) |F(x)|^2 (section 25), so the fraction is
    the integral of F^2 over the core divided by the integral over all x (done here analytically).
    """
    d, h, g = mode["d_um"], mode["h_per_um"], mode["gamma_per_um"]
    if mode["parity"] == "even":
        core = d + np.sin(2 * h * d) / (2 * h)              # integral of cos^2 over [-d, d]
        clad = np.cos(h * d) ** 2 / g                        # two tails, each edge^2/(2 gamma)
    else:
        core = d - np.sin(2 * h * d) / (2 * h)
        clad = np.sin(h * d) ** 2 / g
    return core / (core + clad)


def mode_norm_integral(mode):
    """Integral of F^2 dx over all x (same closed forms as confinement)."""
    d, h, g = mode["d_um"], mode["h_per_um"], mode["gamma_per_um"]
    if mode["parity"] == "even":
        return d + np.sin(2 * h * d) / (2 * h) + np.cos(h * d) ** 2 / g
    return d - np.sin(2 * h * d) / (2 * h) + np.sin(h * d) ** 2 / g


if __name__ == "__main__":
    for t in (0.22, 0.40):
        for m in solve_te_modes(3.50, 1.45, t, 1.31):
            print(t, m["order"], m["parity"], "neff=%.4f" % m["neff"], "1/gamma=%.1f nm" % m["decay_len_nm"],
                  "lambda_g=%.1f nm" % (1e3 * m["lambda_g_um"]), "Gamma=%.3f" % confinement(m), "V=%.3f" % m["V"])
