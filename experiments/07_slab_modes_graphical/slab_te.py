"""Analytic TE modes of the symmetric dielectric slab (docs/NOTES.md sections 14-23).

Geometry (notes section 1): core index n1 for -d < x < d, cladding n2 outside,
propagation along z, E = y F(x) e^{-j beta z}.  Only numpy + scipy, so this file
imports under both the .venv and the .meep interpreters.

Units: lengths in um, k0 = 2 pi / lambda in rad/um, beta in rad/um, gamma in 1/um.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import brentq

MU0 = 1.256_637_06e-6   # H/m
C0 = 299_792_458.0      # m/s


@dataclass
class TEMode:
    m: int            # mode order: 0 = fundamental (even), 1 = first odd, ...
    parity: str       # "even" or "odd"
    lam_um: float
    d_um: float       # half thickness
    n1: float
    n2: float
    V: float
    hd: float         # h*d (dimensionless transverse phase in the core)
    gd: float         # gamma*d
    h_per_um: float
    gamma_per_um: float
    beta_per_um: float
    neff: float
    decay_len_um: float   # 1/gamma
    lambda_g_um: float    # 2 pi / beta = lam / neff
    theta_deg: float      # zig-zag ray angle from the z axis, asin(neff/n1)
    confinement: float    # Gamma_F: fraction of |F|^2 (equivalently of S_z) inside the core
    energy_fraction: float  # Gamma_E: fraction of the electric energy n^2|F|^2 inside the core

    def as_dict(self):
        return asdict(self)


def k0_per_um(lam_um):
    return 2 * np.pi / lam_um


def V_number(d_um, lam_um, n1, n2):
    """Normalised frequency V = k0 d sqrt(n1^2 - n2^2)  (notes 19: the circle radius)."""
    return k0_per_um(lam_um) * d_um * np.sqrt(n1**2 - n2**2)


def num_te_modes(d_um, lam_um, n1, n2):
    """TE_m is guided when V > m pi/2."""
    return int(np.floor(V_number(d_um, lam_um, n1, n2) / (np.pi / 2))) + 1


def cutoff_thickness_um(m, lam_um, n1, n2):
    """Full thickness 2d at which TE_m appears: V = m pi/2."""
    return m * lam_um / (2 * np.sqrt(n1**2 - n2**2))


def cutoff_wavelength_um(m, d_um, n1, n2):
    """Wavelength above which TE_m (m >= 1) is no longer guided at half thickness d."""
    return 4 * d_um * np.sqrt(n1**2 - n2**2) / m


def solve_te_modes(d_um, lam_um, n1, n2):
    """Roots of h tan(hd) = gamma (even) and -h cot(hd) = gamma (odd) on the circle
    (hd)^2 + (gd)^2 = V^2  (notes 19).  Returns modes ordered by m (descending neff)."""
    V = V_number(d_um, lam_um, n1, n2)
    k0 = k0_per_um(lam_um)
    modes = []
    # Branch m lives in hd in (m pi/2, (m+1) pi/2); even for m even, odd for m odd.
    m = 0
    while m * np.pi / 2 < V:
        lo = m * np.pi / 2
        hi = min((m + 1) * np.pi / 2, V)
        if m % 2 == 0:
            def f(u):  # u = hd
                return u * np.tan(u) - np.sqrt(max(V**2 - u**2, 0.0))
        else:
            def f(u):
                return -u / np.tan(u) - np.sqrt(max(V**2 - u**2, 0.0))
        eps = 1e-9
        a, b = lo + eps, hi - eps
        # f(lo+) = -sqrt(V^2 - lo^2) < 0 ; f near the pole (or at u = V) > 0 when the mode exists
        fa, fb = f(a), f(b)
        if fa * fb > 0:  # numerically at cutoff; skip
            m += 1
            continue
        u = brentq(f, a, b, xtol=1e-14, maxiter=500)
        gd = np.sqrt(V**2 - u**2)
        h = u / d_um
        g = gd / d_um
        beta = np.sqrt((n1 * k0)**2 - h**2)
        neff = beta / k0
        parity = "even" if m % 2 == 0 else "odd"
        modes.append(TEMode(
            m=m, parity=parity, lam_um=lam_um, d_um=d_um, n1=n1, n2=n2, V=V,
            hd=u, gd=gd, h_per_um=h, gamma_per_um=g, beta_per_um=beta, neff=neff,
            decay_len_um=1 / g, lambda_g_um=2 * np.pi / beta,
            theta_deg=np.degrees(np.arcsin(neff / n1)),
            confinement=_confinement(u, gd, parity),
            energy_fraction=_energy_fraction(_confinement(u, gd, parity), n1, n2),
        ))
        m += 1
    return modes


def _confinement(hd, gd, parity):
    """Gamma = int_core F^2 / int_all F^2 in closed form, with F(d) = C."""
    if parity == "even":
        core = 1.0 + np.sin(2 * hd) / (2 * hd)      # (1/d) int_-d^d cos^2(hx) dx
        Cd2 = np.cos(hd)**2
    else:
        core = 1.0 - np.sin(2 * hd) / (2 * hd)
        Cd2 = np.sin(hd)**2
    clad = Cd2 / gd                                  # (1/d) 2 int_d^inf C^2 e^{-2g(x-d)} dx
    return core / (core + clad)


def _energy_fraction(gamma_F, n1, n2):
    """Gamma_E = int_core n1^2 F^2 / int_all n^2 F^2, the fraction of the electric energy in the core.
    This (not Gamma_F) is the weight in first-order perturbation theory:
    d(neff)/d(n1) = (n_g/n1) Gamma_E  and  d(neff)/d(n2) = (n_g/n2) (1 - Gamma_E)."""
    core = n1**2 * gamma_F
    clad = n2**2 * (1.0 - gamma_F)
    return core / (core + clad)


def profile(mode: TEMode, x_um):
    """Transverse profile F(x) (normalised to F(0)=1 for even, max slope 1 for odd) and F'(x)."""
    x = np.asarray(x_um, dtype=float)
    d, h, g = mode.d_um, mode.h_per_um, mode.gamma_per_um
    F = np.empty_like(x)
    dF = np.empty_like(x)
    core = np.abs(x) <= d
    right = x > d
    left = x < -d
    if mode.parity == "even":
        F[core] = np.cos(h * x[core]); dF[core] = -h * np.sin(h * x[core])
        C = np.cos(h * d)
        F[right] = C * np.exp(-g * (x[right] - d)); dF[right] = -g * F[right]
        F[left] = C * np.exp(g * (x[left] + d)); dF[left] = g * F[left]
    else:
        F[core] = np.sin(h * x[core]); dF[core] = h * np.cos(h * x[core])
        C = np.sin(h * d)
        F[right] = C * np.exp(-g * (x[right] - d)); dF[right] = -g * F[right]
        F[left] = -C * np.exp(g * (x[left] + d)); dF[left] = g * F[left]   # d/dx[-C e^{g(x+d)}] = g F
    return F, dF


def fields(mode: TEMode, x_um, E0=1.0):
    """Phasor amplitudes at z = 0 (notes 18): E_y = F, H_x = -(beta/(omega mu)) F, H_z = (j/(omega mu)) F'.
    SI: E0 in V/m gives H in A/m.  Returns (Ey, Hx, Hz) with Hz complex."""
    F, dF = profile(mode, x_um)
    omega = 2 * np.pi * C0 / (mode.lam_um * 1e-6)
    beta_si = mode.beta_per_um * 1e6
    Ey = E0 * F
    Hx = -(beta_si / (omega * MU0)) * Ey
    Hz = 1j * (E0 * dF * 1e6) / (omega * MU0)
    return Ey, Hx, Hz


def group_index(d_um, lam_um, n1, n2, m=0, dlam=1e-4):
    """n_g = neff - lam dneff/dlam (notes 22) with fixed material indices
    (waveguide dispersion only; no Sellmeier).  Returns nan if the mode is cut off."""
    def neff_at(l):
        ms = solve_te_modes(d_um, l, n1, n2)
        return ms[m].neff if len(ms) > m else np.nan
    n_plus, n_minus = neff_at(lam_um + dlam), neff_at(lam_um - dlam)
    n0 = neff_at(lam_um)
    return n0 - lam_um * (n_plus - n_minus) / (2 * dlam)
