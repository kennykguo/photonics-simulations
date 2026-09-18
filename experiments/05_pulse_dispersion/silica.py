"""Bulk fused-silica material dispersion from the Malitson (1965) Sellmeier fit.

Everything here is *material* dispersion of bulk SiO2 (notes sections 9-12).
No waveguide contribution is included: a real single-mode fibre adds a
waveguide term that moves the zero-dispersion wavelength from ~1.27 um (bulk)
to ~1.31 um and lowers D(1550) from ~22 to ~17 ps/(nm km).  We say so in the
README; the point of this experiment is the *mechanism*, not a fibre datasheet.

Symbols (SI internally):
    lam    vacuum wavelength [m]        omega  angular frequency [rad/s]
    n      phase index                  n_g    group index = n - lam dn/dlam
    D      material dispersion = -(lam/c) d2n/dlam2   [s/m^2]  (1 ps/(nm km) = 1e-6 s/m^2)
    beta   = n(omega) omega / c  [rad/m]
    beta1  = dbeta/domega   [s/m]       (group delay per metre = n_g / c)
    beta2  = d2beta/domega2 [s^2/m]     (= -lam^2 D / (2 pi c))
    beta3  = d3beta/domega3 [s^3/m]

The derivatives are taken symbolically with sympy on the closed-form Sellmeier
expression, so beta1..beta3 are exact to machine precision (no finite differences).
"""
from __future__ import annotations

import numpy as np
import sympy as sp

C0 = 299_792_458.0  # m/s

# Malitson 1965, lambda in micrometres
B_SELL = (0.6961663, 0.4079426, 0.8974794)
C_SELL_UM2 = (0.0684043**2, 0.1162414**2, 9.896161**2)

# ---- symbolic definitions ---------------------------------------------------
_lam_um = sp.symbols("lambda_um", positive=True)      # wavelength in um
_omega = sp.symbols("omega", positive=True)            # rad/s

_n2_expr = 1 + sum(B * _lam_um**2 / (_lam_um**2 - C) for B, C in zip(B_SELL, C_SELL_UM2))
_n_expr = sp.sqrt(_n2_expr)

# n as a function of omega: lambda[um] = 2 pi c / omega * 1e6
_n_of_omega = _n_expr.subs(_lam_um, 2 * sp.pi * C0 / _omega * 1e6)
_beta_expr = _n_of_omega * _omega / C0                      # rad/m
_beta1_expr = sp.diff(_beta_expr, _omega)
_beta2_expr = sp.diff(_beta_expr, _omega, 2)
_beta3_expr = sp.diff(_beta_expr, _omega, 3)

# wavelength-domain quantities (notes 11): n_g = n - lam dn/dlam, D = -(lam/c) d2n/dlam2
_dn_dlam = sp.diff(_n_expr, _lam_um)              # per um
_d2n_dlam2 = sp.diff(_n_expr, _lam_um, 2)         # per um^2
_ng_expr = _n_expr - _lam_um * _dn_dlam
_D_expr_si = -(_lam_um * 1e-6) / C0 * _d2n_dlam2 * 1e12     # s/m^2  (d2n/dlam2 converted to per m^2)

_f_n = sp.lambdify(_lam_um, _n_expr, "numpy")
_f_ng = sp.lambdify(_lam_um, _ng_expr, "numpy")
_f_D = sp.lambdify(_lam_um, _D_expr_si, "numpy")
_f_d2n = sp.lambdify(_lam_um, _d2n_dlam2, "numpy")
_f_beta = sp.lambdify(_omega, _beta_expr, "numpy")
_f_beta1 = sp.lambdify(_omega, _beta1_expr, "numpy")
_f_beta2 = sp.lambdify(_omega, _beta2_expr, "numpy")
_f_beta3 = sp.lambdify(_omega, _beta3_expr, "numpy")


def n(lam_m):
    """Phase index at vacuum wavelength lam [m]."""
    return _f_n(np.asarray(lam_m) * 1e6)


def n_g(lam_m):
    """Group index n_g = n - lam dn/dlam (notes 11)."""
    return _f_ng(np.asarray(lam_m) * 1e6)


def D(lam_m):
    """Material dispersion parameter D = -(lam/c) d2n/dlam2 in SI [s/m^2]."""
    return _f_D(np.asarray(lam_m) * 1e6)


def D_ps_nm_km(lam_m):
    return D(lam_m) * 1e6


def beta(omega):
    """Exact propagation constant beta(omega) = n(omega) omega / c [rad/m]."""
    return _f_beta(np.asarray(omega, dtype=float))


def beta1(omega):
    return _f_beta1(float(omega))


def beta2(omega):
    return _f_beta2(float(omega))


def beta3(omega):
    return _f_beta3(float(omega))


def omega_of_lam(lam_m):
    return 2 * np.pi * C0 / lam_m


def zero_dispersion_wavelength():
    """Root of D(lam) between 1.1 and 1.5 um, found by bisection on the exact expression."""
    from scipy.optimize import brentq
    return brentq(lambda l: D(l), 1.1e-6, 1.5e-6)


def symbolic_summary() -> str:
    """Return the symbolic derivative chain as text (for the README / results.txt)."""
    lines = [
        "n^2(lambda) - 1 = sum_i B_i lambda^2 / (lambda^2 - C_i)   [Malitson fused silica, lambda in um]",
        f"  B = {B_SELL}",
        f"  C = {tuple(round(c, 6) for c in C_SELL_UM2)} um^2",
        "beta(omega)  = n(omega) omega / c,  omega = 2 pi c / lambda",
        "beta1 = d beta / d omega = n_g / c,          n_g = n - lambda dn/dlambda",
        "beta2 = d^2 beta / d omega^2 = -lambda^2 D / (2 pi c),   D = -(lambda/c) d^2 n / d lambda^2",
        "beta3 = d^3 beta / d omega^3   (kept in the exact propagator, shown to be negligible)",
    ]
    return "\n".join(lines)
