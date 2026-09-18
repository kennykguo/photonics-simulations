"""Sellmeier material models and the symbolic/numeric dispersion toolkit.

Implements notes §9 (Sellmeier form), §10 (UV vs IR curvature), §11 (n_g and D).

    n²(λ) − 1 = Σ_i B_i λ² / (λ² − C_i)          (λ in µm, C_i in µm²)
    n_g = n − λ dn/dλ
    D   = −(λ/c) d²n/dλ²        (SI s/m², reported in ps/(nm·km))

Everything numeric is *lambdified from the sympy expressions*, so the numbers in
run.py are literally the evaluated symbolic derivation, not a separate hand-coded formula.
A finite-difference cross-check lives in run.py.
"""
from __future__ import annotations

import sympy as sp
import numpy as np

C0 = 299_792_458.0  # m/s

# --------------------------------------------------------------------------
# Coefficient tables (λ in µm)
# --------------------------------------------------------------------------
# Malitson 1965, fused silica, valid 0.21-3.71 µm
SILICA_B = (0.6961663, 0.4079426, 0.8974794)
SILICA_C = (0.0684043**2, 0.1162414**2, 9.896161**2)   # µm²  (√C = resonance λ: 68 nm, 116 nm, 9.9 µm)

# Salzberg & Villa 1957, crystalline silicon, fitted 1.36-11 µm (used here ≥ 1.2 µm)
#   n² = A + B/λ² + C·λ1²/(λ² − λ1²)
SI_A, SI_B, SI_C, SI_L1 = 11.6858, 0.939816, 0.00810461, 1.1071


# --------------------------------------------------------------------------
# Symbolic model
# --------------------------------------------------------------------------
lam = sp.symbols("lambda", positive=True)          # vacuum wavelength, µm
c_sym = sp.symbols("c", positive=True)             # speed of light (symbolic in the derivation)
B1, B2, B3, C1, C2, C3 = sp.symbols("B1 B2 B3 C1 C2 C3", positive=True)


def sellmeier_n2_generic():
    """n² for a generic three-term Sellmeier (symbolic coefficients)."""
    return 1 + B1 * lam**2 / (lam**2 - C1) + B2 * lam**2 / (lam**2 - C2) + B3 * lam**2 / (lam**2 - C3)


def dispersion_expressions(n_expr):
    """Given a symbolic n(λ), return (n, dn/dλ, n_g, d²n/dλ², D) as sympy expressions.

    D is left with the symbol c; convert to ps/(nm·km) with D_ps_nm_km().
    """
    dn = sp.diff(n_expr, lam)
    d2n = sp.diff(n_expr, lam, 2)
    ng = n_expr - lam * dn                 # notes §11
    D = -(lam / c_sym) * d2n               # notes §11, SI: s/m² if λ in m
    return n_expr, dn, ng, d2n, D


def silica_n_expr():
    return sp.sqrt(sellmeier_n2_generic().subs({B1: SILICA_B[0], B2: SILICA_B[1], B3: SILICA_B[2],
                                                C1: SILICA_C[0], C2: SILICA_C[1], C3: SILICA_C[2]}))


def silicon_n_expr():
    return sp.sqrt(SI_A + SI_B / lam**2 + SI_C * SI_L1**2 / (lam**2 - SI_L1**2))


# --------------------------------------------------------------------------
# Numeric wrappers (lambdified from the symbolic forms)
# --------------------------------------------------------------------------
class Material:
    """Numeric n, dn/dλ, n_g, d²n/dλ², D for one Sellmeier material, all from sympy."""

    def __init__(self, name: str, n_expr, valid_um: tuple[float, float]):
        self.name = name
        self.valid_um = valid_um
        self.n_expr, self.dn_expr, self.ng_expr, self.d2n_expr, self.D_expr = dispersion_expressions(n_expr)
        self._n = sp.lambdify(lam, self.n_expr, "numpy")
        self._dn = sp.lambdify(lam, self.dn_expr, "numpy")
        self._ng = sp.lambdify(lam, self.ng_expr, "numpy")
        self._d2n = sp.lambdify(lam, self.d2n_expr, "numpy")

    def n(self, lam_um):
        return self._n(np.asarray(lam_um, dtype=float))

    def dn_dlam(self, lam_um):
        """dn/dλ in 1/µm."""
        return self._dn(np.asarray(lam_um, dtype=float))

    def ng(self, lam_um):
        return self._ng(np.asarray(lam_um, dtype=float))

    def d2n_dlam2(self, lam_um):
        """d²n/dλ² in 1/µm²."""
        return self._d2n(np.asarray(lam_um, dtype=float))

    def D_ps_nm_km(self, lam_um):
        """Material dispersion D = −(λ/c) d²n/dλ² in ps/(nm·km)."""
        return D_ps_nm_km(np.asarray(lam_um, dtype=float), self.d2n_dlam2(lam_um))

    def beta2_ps2_km(self, lam_um):
        """β₂ = −λ²D/(2πc) in ps²/km (notes §12)."""
        lam_m = np.asarray(lam_um, dtype=float) * 1e-6
        D_si = self.D_ps_nm_km(lam_um) * 1e-12 / 1e-9 / 1e3       # s/m²
        return -(lam_m**2) * D_si / (2 * np.pi * C0) * 1e24 / 1e-3  # ps²/km


def D_ps_nm_km(lam_um, d2n_per_um2):
    """Convert λ [µm] and d²n/dλ² [1/µm²] into D [ps/(nm·km)].

    D = −(λ/c) d²n/dλ²  with λ in m and d²n/dλ² in 1/m²  → s/m².
    1 s/m² = 1e12 ps / (1e9 nm · 1e-3 km) = 1e6 ps/(nm·km).
    """
    lam_m = np.asarray(lam_um, dtype=float) * 1e-6
    d2n_m = np.asarray(d2n_per_um2, dtype=float) * 1e12       # 1/µm² → 1/m²
    return -(lam_m / C0) * d2n_m * 1e6


def silica() -> Material:
    return Material("fused silica (Malitson 1965)", silica_n_expr(), (0.21, 3.71))


def silicon() -> Material:
    return Material("crystalline silicon (Salzberg-Villa 1957)", silicon_n_expr(), (1.2, 11.0))


# --------------------------------------------------------------------------
# Term-by-term Sellmeier contributions (notes §10)
# --------------------------------------------------------------------------
def silica_terms(lam_um):
    """Return list of (label, S_i, S_i', S_i'') for each Sellmeier term of silica, in µm units."""
    out = []
    lam_arr = np.asarray(lam_um, dtype=float)
    for i, (B, C) in enumerate(zip(SILICA_B, SILICA_C)):
        S = B * lam**2 / (lam**2 - C)
        S1 = sp.diff(S, lam)
        S2 = sp.diff(S, lam, 2)
        label = f"term {i+1}: √C = {np.sqrt(C)*1e3:.0f} nm ({'UV' if np.sqrt(C) < 0.5 else 'IR'})"
        out.append((label,
                    sp.lambdify(lam, S, "numpy")(lam_arr),
                    sp.lambdify(lam, S1, "numpy")(lam_arr),
                    sp.lambdify(lam, S2, "numpy")(lam_arr)))
    return out


def small_change_rule_ghz_per_nm(lam_nm: float) -> float:
    """|df/dλ| = c/λ² expressed in GHz per nm (notes: 1 nm ↔ 174.7 GHz at 1310 nm)."""
    lam_m = lam_nm * 1e-9
    return C0 / lam_m**2 * 1e-9 * 1e-9   # Hz/m → GHz/nm
