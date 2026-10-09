"""Effective index and group index of a silicon waveguide, computed not assumed.

Two numbers describe how fast light goes down a waveguide, and they are not the
same number:

    n_eff = beta / k0                        the PHASE index.
                                             Wave crests move at c / n_eff.

    n_g   = c dbeta/domega                   the GROUP index.
          = n_eff - lambda dn_eff/dlambda    The envelope, and with it the energy
                                             and the information, moves at c / n_g.

For the silicon-on-insulator strip in the Lightmatter capstone (500 nm wide,
220 nm thick, silica cladding, 1310 nm) they come out about 2.71 and 4.16: the
crests travel roughly 1.5 times faster than the pulse they belong to.

Lengths are in micrometres throughout, the usual photonics convention, so
lambda = 1.31 means 1310 nm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

C0 = 299_792_458.0  # m/s


# --------------------------------------------------------------------------
# Material dispersion (Sellmeier).
#
# Both glass and silicon slow blue light more than red, and that material
# dispersion is part of the group index. Using real fits rather than fixed
# numbers is what makes n_g come out right.
# --------------------------------------------------------------------------


def n_silica(lam_um):
    """Fused silica, Malitson 1965. n(1.31 um) = 1.4468."""
    l2 = np.asarray(lam_um, dtype=float) ** 2
    n2 = (
        1.0
        + 0.6961663 * l2 / (l2 - 0.0684043**2)
        + 0.4079426 * l2 / (l2 - 0.1162414**2)
        + 0.8974794 * l2 / (l2 - 9.896161**2)
    )
    return np.sqrt(n2)


def n_silicon(lam_um):
    """Crystalline silicon, Salzberg and Villa. n(1.31 um) = 3.5005."""
    l2 = np.asarray(lam_um, dtype=float) ** 2
    n2 = 11.6858 + 0.939816 / l2 + 0.00810461 * 1.1071**2 / (l2 - 1.1071**2)
    return np.sqrt(n2)


def bulk_group_index(n_of_lambda, lam_um: float, h: float = 2e-4) -> float:
    """n_g = n - lambda dn/dlambda for a bulk material (no waveguide)."""
    dn = (float(n_of_lambda(lam_um + h)) - float(n_of_lambda(lam_um - h))) / (2 * h)
    return float(n_of_lambda(lam_um)) - lam_um * dn


# --------------------------------------------------------------------------
# The symmetric slab: the one geometry whose eigenvalue equation is closed form,
# so it can be solved exactly and used to check everything else.
#
#   even modes:   h tan(h d) = gamma
#   odd modes:   -h cot(h d) = gamma
#   h = k0 sqrt(n_core^2 - n_eff^2)      transverse wavenumber in the core
#   gamma = k0 sqrt(n_eff^2 - n_clad^2)  decay rate in the cladding
#
# Both families are one equation, h d - arctan(gamma/h) = m pi/2, monotonic in
# n_eff, so bisection between the cladding and core indices always finds it.
# --------------------------------------------------------------------------


def slab_neff(lam_um: float, thickness_um: float, n_core: float, n_clad: float,
              mode: int = 0):
    """TE mode `mode` of a symmetric slab. Returns None if it is below cutoff."""
    k0 = 2 * math.pi / lam_um
    d = thickness_um / 2.0

    def residual(neff: float) -> float:
        h = k0 * math.sqrt(max(n_core**2 - neff**2, 1e-15))
        gamma = k0 * math.sqrt(max(neff**2 - n_clad**2, 1e-15))
        return h * d - math.atan2(gamma, h) - mode * math.pi / 2.0

    lo, hi = n_clad + 1e-9, n_core - 1e-9
    if residual(lo) * residual(hi) > 0:
        return None
    return float(brentq(residual, lo, hi, xtol=1e-12, rtol=1e-14))


def slab_profile(lam_um: float, thickness_um: float, n_core: float, n_clad: float,
                 neff: float, x_um: np.ndarray) -> np.ndarray:
    """Field F(x) of the fundamental (even) slab mode: cosine inside, decaying outside."""
    k0 = 2 * math.pi / lam_um
    d = thickness_um / 2.0
    h = k0 * math.sqrt(n_core**2 - neff**2)
    gamma = k0 * math.sqrt(neff**2 - n_clad**2)
    f = np.where(
        np.abs(x_um) <= d,
        np.cos(h * np.clip(x_um, -d, d)),
        math.cos(h * d) * np.exp(-gamma * (np.abs(x_um) - d)),
    )
    return f / np.abs(f).max()


# --------------------------------------------------------------------------
# The real thing: a rectangular strip has no closed form, so it needs a
# numerical mode solver. femwell solves the full vector Maxwell eigenproblem
# with finite elements on a triangular mesh.
# --------------------------------------------------------------------------


@dataclass
class StripMode:
    n_eff: float
    te_fraction: float
    dneff_dnsi: float          # how much n_eff moves per unit change of the silicon
                               # index. femwell's confinement functional over the core
                               # equals exactly this, verified against a direct re-solve.
    energy_fraction_core: float  # sqrt(eps)-weighted energy fraction inside the silicon
    mode_obj: object           # the femwell Mode, kept so it can plot itself


def strip_modes(lam_um: float, width_um: float, height_um: float,
                n_core: float | None = None, n_clad: float | None = None,
                resolution: float = 0.04, pad_um: float = 0.8,
                num_modes: int = 1):
    """Full-vector FEM mode solve of a rectangular strip in a uniform cladding.

    Materials default to the Sellmeier values at this wavelength, so a sweep in
    wavelength automatically carries the material dispersion with it.
    """
    from collections import OrderedDict

    import shapely
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio

    n_core = float(n_silicon(lam_um)) if n_core is None else n_core
    n_clad = float(n_silica(lam_um)) if n_clad is None else n_clad

    core = shapely.box(-width_um / 2, -height_um / 2, width_um / 2, height_um / 2)
    clad = shapely.box(-width_um / 2 - pad_um, -height_um / 2 - pad_um,
                       width_um / 2 + pad_um, height_um / 2 + pad_um)
    mesh = from_meshio(
        mesh_from_OrderedDict(
            OrderedDict(core=core, clad=clad),
            {"core": {"resolution": resolution, "distance": 0.4}},
            default_resolution_max=0.2,
        )
    )
    basis0 = Basis(mesh, ElementTriP0())
    eps = basis0.zeros()
    for sub, n in {"core": n_core, "clad": n_clad}.items():
        eps[basis0.get_dofs(elements=sub)] = n**2

    out = []
    for m in compute_modes(basis0, eps, wavelength=lam_um,
                           num_modes=num_modes, order=2):
        try:
            s_core = float(np.real(m.calculate_confinement_factor("core")))
            s_clad = float(np.real(m.calculate_confinement_factor("clad")))
            frac = s_core / (s_core + s_clad)
        except Exception:
            s_core, frac = float("nan"), float("nan")
        out.append(
            StripMode(
                n_eff=float(np.real(m.n_eff)),
                te_fraction=float(np.real(m.te_fraction)),
                dneff_dnsi=s_core,
                energy_fraction_core=frac,
                mode_obj=m,
            )
        )
    return out


def neff_strip(lam_um: float, width_um: float, height_um: float, **kw) -> float:
    """Just the fundamental mode's effective index."""
    modes = strip_modes(lam_um, width_um, height_um, **kw)
    return modes[0].n_eff if modes else float("nan")


# --------------------------------------------------------------------------
# Group index: differentiate n_eff(lambda) numerically, letting the material
# indices move with wavelength too, which makes this the TOTAL group index
# (waveguide dispersion plus material dispersion) rather than either alone.
# --------------------------------------------------------------------------


def group_index_strip(lam_um: float, width_um: float, height_um: float,
                      dlam: float = 0.02, **kw):
    """Returns (n_eff, n_g, dn_eff/dlambda) for the strip's fundamental mode."""
    n0 = neff_strip(lam_um, width_um, height_um, **kw)
    nm = neff_strip(lam_um - dlam, width_um, height_um, **kw)
    np_ = neff_strip(lam_um + dlam, width_um, height_um, **kw)
    slope = (np_ - nm) / (2 * dlam)
    return n0, n0 - lam_um * slope, slope


def group_index_slab(lam_um: float, thickness_um: float, dlam: float = 0.02,
                     mode: int = 0):
    """Same, for the exact slab. Cheap enough to call thousands of times."""
    def f(l):
        return slab_neff(l, thickness_um, float(n_silicon(l)), float(n_silica(l)), mode)

    n0, nm, np_ = f(lam_um), f(lam_um - dlam), f(lam_um + dlam)
    if None in (n0, nm, np_):
        return None, None, None
    slope = (np_ - nm) / (2 * dlam)
    return n0, n0 - lam_um * slope, slope


def velocities(n_eff: float, n_g: float, lam_um: float = 1.31) -> dict:
    """Everything the two indices mean physically, in units you can picture."""
    return {
        "v_phase_m_per_s": C0 / n_eff,
        "v_group_m_per_s": C0 / n_g,
        "v_phase_um_per_ps": C0 / n_eff * 1e-6,
        "v_group_um_per_ps": C0 / n_g * 1e-6,
        "ratio_vp_over_vg": n_g / n_eff,
        "guided_wavelength_nm": lam_um / n_eff * 1000,
        "optical_period_fs": lam_um * 1e-6 / C0 * 1e15,
    }
