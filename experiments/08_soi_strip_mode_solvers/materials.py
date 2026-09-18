"""Bulk refractive indices used by experiment 08 (Sellmeier fits, wavelengths in um).

Silicon: Salzberg & Villa 1957 (same fit as experiment 04):
    n^2 = 11.6858 + 0.939816/lam^2 + 0.00810461*1.1071^2/(lam^2 - 1.1071^2)
Fused silica: Malitson 1965 three-term Sellmeier.
At 1.310 um these give n_Si = 3.5006 and n_SiO2 = 1.4468, i.e. the rounded REF
values 3.50 and 1.45 used everywhere else in the suite.  The *slope* dn/dlam is
what matters here: it is the material-dispersion part of the group index
(notes section 11: n_g = n - lam dn/dlam).
"""
import numpy as np


def n_si(lam_um):
    lam2 = np.asarray(lam_um, dtype=float) ** 2
    n2 = 11.6858 + 0.939816 / lam2 + 0.00810461 * 1.1071**2 / (lam2 - 1.1071**2)
    return np.sqrt(n2)


def n_sio2(lam_um):
    lam2 = np.asarray(lam_um, dtype=float) ** 2
    B = (0.6961663, 0.4079426, 0.8974794)
    C = (0.0684043**2, 0.1162414**2, 9.896161**2)
    n2 = 1.0 + sum(b * lam2 / (lam2 - c) for b, c in zip(B, C))
    return np.sqrt(n2)


def group_index(n_func, lam_um, dlam=1e-4):
    """n_g = n - lam dn/dlam by central finite difference (notes 11)."""
    lam = np.asarray(lam_um, dtype=float)
    dn = (n_func(lam + dlam) - n_func(lam - dlam)) / (2 * dlam)
    return n_func(lam) - lam * dn


if __name__ == "__main__":
    for lam in (1.26, 1.31, 1.36, 1.55):
        print(f"lam={lam:.3f} um  n_si={n_si(lam):.4f} ng_si={group_index(n_si, lam):.4f}  "
              f"n_sio2={n_sio2(lam):.4f} ng_sio2={group_index(n_sio2, lam):.4f}")
