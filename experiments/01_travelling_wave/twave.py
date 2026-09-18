"""Physics helpers for experiment 01: the travelling wave E(z,t) = E0 cos(wt - bz).

Everything here is SI internally (m, s, rad/m, rad/s). The plotting code converts
to nm and fs for display. Convention (NOTES 2, 3): e^{jwt} time dependence, a +z wave
carries e^{-jbz}, so the physical field is Re{E0 e^{-jbz} e^{jwt}} = E0 cos(wt - bz).
"""
from __future__ import annotations

import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common import REF, C0  # noqa: E402


# ---------------------------------------------------------------- numbers
def wave_numbers(lambda_nm: float = REF.lambda_nm, n: float = REF.neff) -> dict:
    """All the scalar numbers of a monochromatic wave at vacuum wavelength lambda_nm
    travelling in a medium (or mode) of index n. NOTES 2, 4, 22."""
    lam0 = lambda_nm * 1e-9
    f = C0 / lam0
    omega = 2 * np.pi * f
    T = 1 / f
    k0 = 2 * np.pi / lam0
    beta = n * k0
    lam_g = lam0 / n
    v_p = omega / beta
    return dict(
        lambda0_m=lam0, f_Hz=f, omega_rad_s=omega, T_s=T, k0_rad_m=k0,
        n=n, beta_rad_m=beta, lambda_med_m=lam_g, v_p_m_s=v_p,
    )


# ---------------------------------------------------------------- fields
def E_real(z, t, omega, beta, sign=-1, E0=1.0):
    """Physical field E0 cos(wt + sign*bz). sign=-1 -> +z travelling (NOTES 2)."""
    return E0 * np.cos(omega * t + sign * beta * z)


def E_phasor(z, beta, sign=-1, E0=1.0):
    """Spatial phasor E~(z) = E0 exp(j*sign*bz); sign=-1 gives e^{-jbz} (NOTES 3)."""
    return E0 * np.exp(1j * sign * beta * z)


def E_from_phasor(z, t, omega, beta, sign=-1, E0=1.0):
    """Re{E~(z) e^{jwt}}: must equal E_real exactly."""
    return np.real(E_phasor(z, beta, sign, E0) * np.exp(1j * omega * t))


# ---------------------------------------------------------------- crest tracking
def _parabolic_peak(z, y, i):
    """Sub-grid peak position from a parabola through points i-1, i, i+1."""
    if i <= 0 or i >= len(z) - 1:
        return z[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return z[i]
    dz = z[1] - z[0]
    return z[i] + 0.5 * dz * (y0 - y2) / denom


def track_crest(z, t, omega, beta, sign=-1, z_start=None):
    """Follow ONE crest of E(z,t) through time by continuity (nearest local maximum
    to the previous crest position) and return its position z_c(t).

    This is the numerical version of 'following a crest, wt - bz = phi0' (NOTES 2);
    the slope of z_c(t) is the phase velocity and must equal w/b.
    """
    if z_start is None:
        # at t=0 the field is cos(sign*bz): maximum at z=0 -> start at the crest closest
        # to the middle of the window so it does not leave the grid immediately
        n_lam = 2 * np.pi / beta
        z_start = np.round(np.mean(z) / n_lam) * n_lam
    zc = np.empty_like(t)
    prev = z_start
    for i, ti in enumerate(t):
        y = E_real(z, ti, omega, beta, sign)
        # local maxima
        idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
        if len(idx) == 0:
            zc[i] = np.nan
            continue
        j = idx[np.argmin(np.abs(z[idx] - prev))]
        prev = _parabolic_peak(z, y, j)
        zc[i] = prev
    return zc


def fit_velocity(t, zc):
    """Least-squares slope dz/dt of the tracked crest (m/s) and its intercept."""
    ok = np.isfinite(zc)
    slope, intercept = np.polyfit(t[ok], zc[ok], 1)
    return slope, intercept


# ---------------------------------------------------------------- illustrative mode profile
def slab_profile_illustrative(x, lambda_nm=REF.lambda_nm, n1=REF.n_si, n2=REF.n_sio2,
                              neff=REF.neff, half_thickness_m=REF.wg_height_um * 1e-6 / 2):
    """cos(hx) inside |x|<d, exponential outside, with h and gamma taken from the
    reference n_eff (NOTES 16). Illustrative only: n_eff = 2.5 is not the true
    eigenvalue of the 220 nm slab, so F' is not continuous at x = +-d; experiment 07
    solves the real eigenproblem."""
    k0 = 2 * np.pi / (lambda_nm * 1e-9)
    h = k0 * np.sqrt(n1**2 - neff**2)
    g = k0 * np.sqrt(neff**2 - n2**2)
    d = half_thickness_m
    F = np.where(np.abs(x) <= d, np.cos(h * x), np.cos(h * d) * np.exp(-g * (np.abs(x) - d)))
    return F, h, g


# ---------------------------------------------------------------- capstone: one round trip
def neff_of_lambda(lambda_nm, neff=REF.neff, ng=REF.ng, lambda0_nm=REF.lambda_nm):
    """First-order dispersion of the effective index about lambda0 (NOTES 22):
    n_g = n_eff - lambda dn_eff/dlambda  ->  dn_eff/dlambda = (n_eff - n_g)/lambda0."""
    return neff - (np.asarray(lambda_nm) - lambda0_nm) * (ng - neff) / lambda0_nm


def round_trip_phase(lambda_nm, L_um=REF.round_trip_um, neff=REF.neff, ng=REF.ng,
                     lambda0_nm=REF.lambda_nm, dispersive=True):
    """Phase accumulated in one lap of the ring, beta*L = 2*pi*n_eff(lambda)*L/lambda (rad).
    The round-trip factor of the ring transfer function is e^{-j beta L} (NOTES 3, 28).
    dispersive=False freezes n_eff at its 1310 nm value (phase index only)."""
    lam = np.asarray(lambda_nm, dtype=float)
    n = neff_of_lambda(lam, neff, ng, lambda0_nm) if dispersive else neff
    return 2 * np.pi * n * (L_um * 1e3) / lam
