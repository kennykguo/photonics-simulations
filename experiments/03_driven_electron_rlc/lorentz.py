"""Analytic Lorentz-oscillator / series-RLC helpers shared by run.py and explore.ipynb.

Conventions (docs/NOTES.md, sections 6-8): phasors e^{jwt}; n = n' - j n'' with n'' > 0 for loss.

Mechanical oscillator (one bound electron, notes section 7):
    m x'' + m*gamma x' + m*w0^2 x = -q E(t)
Series RLC driven by a voltage source (charge q on the capacitor is the state):
    L q'' + R q' + (1/C) q = V(t)
Term-by-term map:  m <-> L,  m*gamma <-> R  (gamma = R/L),  m*w0^2 <-> 1/C  (w0^2 = 1/(LC)),
                   -q_e E <-> V,  x <-> q_C.
Because the dipole is p = -q_e x, the capacitor charge q_C plays the role of the *dipole* (or the
polarisation P = N p), which is in phase with E far below resonance. That is why the transfer
function q_C/V of the circuit has exactly the shape of chi(w) = A/(w0^2 - w^2 + j*gamma*w).
"""
import numpy as np

C0 = 299_792_458.0


def chi_hat(Omega, Gamma):
    """Dimensionless susceptibility chi(w)*w0^2/A = 1/(1 - Omega^2 + j*Gamma*Omega).

    Omega = w/w0, Gamma = gamma/w0 = 1/Q.  Real part = in-phase response, -Imag part = lossy part.
    """
    Omega = np.asarray(Omega, dtype=float)
    return 1.0 / (1.0 - Omega**2 + 1j * Gamma * Omega)


def phase_lag_deg(Omega, Gamma):
    """Phase by which the displacement (or capacitor charge) LAGS the drive, in degrees (0..180)."""
    Omega = np.asarray(Omega, dtype=float)
    return np.degrees(np.arctan2(Gamma * Omega, 1.0 - Omega**2))


def rlc_transfer(f, L, R, C):
    """Charge-per-volt transfer function H(w) = Q_C / V of the series RLC (SI units, complex)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    return 1.0 / (L * (1.0 / (L * C) - w**2 + 1j * (R / L) * w))


def rlc_absorbed_power(f, L, R, C, V0):
    """Time-averaged power absorbed by the resistor for a drive of amplitude V0 (peak volts)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    Z = R + 1j * (w * L - 1.0 / (w * C))
    return 0.5 * V0**2 * R / np.abs(Z) ** 2


def rlc_values(f0_hz, Q, L):
    """Return (L, R, C, w0, gamma) for a series RLC with resonance f0 and quality factor Q."""
    w0 = 2 * np.pi * f0_hz
    C = 1.0 / (w0**2 * L)
    gamma = w0 / Q
    R = gamma * L
    return L, R, C, w0, gamma


def n_complex_from_chi(chi):
    """Complex index n = n' - j n'' from chi = chi' - j chi''  (n^2 = 1 + chi, notes section 6).

    The square-root branch is chosen so that n' > 0 (the passive branch)."""
    n = np.sqrt(1.0 + np.asarray(chi, dtype=complex))
    n = np.where(n.real < 0, -n, n)
    return n


def lorentz_sum(omega, A_list, w_list, g_list):
    """chi(w) = sum_i A_i / (w_i^2 - w^2 + j g_i w)   (notes section 7, several oscillators add)."""
    omega = np.asarray(omega, dtype=float)
    chi = np.zeros_like(omega, dtype=complex)
    for A, wi, gi in zip(A_list, w_list, g_list):
        chi = chi + A / (wi**2 - omega**2 + 1j * gi * omega)
    return chi


# Malitson (1965) fused-silica Sellmeier coefficients, lambda in um (notes section 9: B_i = A_i lam_i^2/(2 pi c)^2)
MALITSON_B = (0.6961663, 0.4079426, 0.8974794)
MALITSON_LAM_UM = (0.0684043, 0.1162414, 9.896161)


def malitson_oscillators():
    """Convert the Sellmeier table to Lorentz-oscillator strengths A_i (s^-2) and resonances w_i (rad/s)."""
    w_i = np.array([2 * np.pi * C0 / (lam * 1e-6) for lam in MALITSON_LAM_UM])
    A_i = np.array(MALITSON_B) * w_i**2          # B_i = A_i / w_i^2  (from B_i = A_i lam_i^2/(2 pi c)^2)
    return A_i, w_i


def sellmeier_n(lam_um, B=MALITSON_B, lam_res_um=MALITSON_LAM_UM):
    lam2 = np.asarray(lam_um, dtype=float) ** 2
    s = np.zeros_like(lam2)
    for b, lr in zip(B, lam_res_um):
        s = s + b * lam2 / (lam2 - lr**2)
    return np.sqrt(1.0 + s)


def effective_uv_oscillator():
    """Collapse the two UV Sellmeier terms into one 'effective' UV resonance (strength-weighted lam^2)."""
    B1, B2 = MALITSON_B[:2]
    l1, l2 = MALITSON_LAM_UM[:2]
    B_u = B1 + B2
    lam_u = np.sqrt((B1 * l1**2 + B2 * l2**2) / B_u)
    w_u = 2 * np.pi * C0 / (lam_u * 1e-6)
    return B_u, lam_u, w_u


# ---------------------------------------------------------------------------------------------------
# Free carriers: the Lorentz oscillator with the spring removed (w0 -> 0) is the Drude model.
# ---------------------------------------------------------------------------------------------------
def drude_chi_hat(Omega, Gamma):
    """Dimensionless free-carrier susceptibility -1/(Omega^2 - j*Gamma*Omega), same normalisation as chi_hat.

    Omega = w/w_s where w_s is only a frequency scale (there is no resonance); Gamma = (1/tau)/w_s.
    Re part is NEGATIVE: free carriers LOWER the refractive index; -Im part > 0 is free-carrier absorption."""
    Omega = np.asarray(Omega, dtype=float)
    return -1.0 / (Omega**2 - 1j * Gamma * Omega)


def drude_silicon(dN_cm3, lam_m, n_host, m_eff_kg, tau_s):
    """Drude change of the complex index of silicon for dN electrons per cm^3 (SI inside).

    chi_fc = -w_p^2 / (w^2 - j w/tau),  w_p^2 = N q^2/(eps0 m*);  small perturbation: dn = chi_fc/(2 n).
    Returns (dn_real, dn_imag_positive, dalpha_per_cm)."""
    Q_E = 1.602_176_634e-19; EPS0 = 8.854_187_8e-12
    N = np.asarray(dN_cm3, dtype=float) * 1e6
    w = 2 * np.pi * C0 / lam_m
    wp2 = N * Q_E**2 / (EPS0 * m_eff_kg)
    chi = -wp2 / (w**2 - 1j * w / tau_s)
    dn = chi / (2 * n_host)                  # n = sqrt(n0^2 + chi) ~ n0 + chi/(2 n0)
    npp = -dn.imag
    alpha_m = 4 * np.pi * npp / lam_m
    return dn.real, npp, alpha_m / 100.0


def soref_bennett_1310(dN_cm3):
    """Empirical Soref & Bennett (1987) coefficients at 1.3 um for ELECTRONS: dn and dalpha (1/cm)."""
    N = np.asarray(dN_cm3, dtype=float)
    return -6.2e-22 * N, 6.0e-18 * N


def sellmeier_dn_dlam_terms(lam_um, B=MALITSON_B, lam_res_um=MALITSON_LAM_UM):
    """dn/dlambda (1/um) split into the contribution of each Sellmeier resonance (notes section 10).

    S_i = B_i lam^2/(lam^2 - C_i);  dS_i/dlam = -2 B_i lam C_i/(lam^2 - C_i)^2;  dn/dlam = (sum dS_i/dlam)/(2n)."""
    lam = float(lam_um)
    n = float(sellmeier_n(lam, B, lam_res_um))
    terms = []
    for b, lr in zip(B, lam_res_um):
        C = lr**2
        terms.append(-2 * b * lam * C / (lam**2 - C) ** 2 / (2 * n))
    return np.array(terms), n
