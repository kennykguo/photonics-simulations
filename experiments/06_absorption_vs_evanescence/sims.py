"""Meep runs for experiment 06: absorption (decay along the flow) vs evanescence (decay across it).

Two tiny FDTD problems, both solved for their steady-state phasors with a narrow-band
Gaussian pulse + DFT monitors (a DFT of a continuous source would carry the turn-on transient
and show a standing-wave ripple; a pulse that has fully decayed gives the clean phasor).

Convention bridge
-----------------
Meep integrates real-time fields with the physicist's e^{-iωt} convention; the notes use the
engineering e^{+jωt}.  A Meep DFT phasor A_m satisfies E(t) = Re{A_m e^{-iωt}} = Re{conj(A_m) e^{+jωt}},
so every phasor returned here is conj(Meep DFT) and obeys the notes: +z wave = e^{-jβz},
loss = n' - j n'', evanescent tail = e^{-γx}.

Meep units: lengths in µm, frequency f = 1/λ[µm], ε0 = μ0 = c = 1, so S = E × H and
p_abs = ½ ω ε'' |E|² with ω = 2π f can be compared directly (both in the same arbitrary unit).

Axis bridge for the 2-D run (Meep's 2-D cell is x-y with Ez out of plane):
  notes' z (propagation, along the interface)  = Meep x
  notes' x (normal to the interface)           = Meep y
  notes' y (direction of E for the TE field)   = Meep z
"""
import numpy as np
import meep as mp

LAM = 1.31                     # µm
F = 1.0 / LAM                  # Meep frequency
OMEGA = 2 * np.pi * F          # Meep angular frequency (c = 1)


def meep_lossy_medium(n_re: float, n_im: float) -> mp.Medium:
    """Medium with complex index n = n' - j n'' (engineering convention).

    ε = n² = (n'² - n''²) - j 2 n' n''.  Meep expresses loss as a conductivity σ_D with
    ε(ω) = ε_inf (1 + i σ_D / ω) in its e^{-iωt} convention, so σ_D = ω ε''/ε_inf.
    """
    eps_inf = n_re ** 2 - n_im ** 2
    eps_im = 2 * n_re * n_im
    return mp.Medium(epsilon=eps_inf, D_conductivity=OMEGA * eps_im / eps_inf)


def run_lossy_plane_wave(n_re: float, n_im: float, length_um: float, resolution: int = 200,
                         pml_um: float = 1.0, source_offset_um: float = 0.5):
    """1-D FDTD: x-polarised plane wave launched at the left, travelling +z through a uniform
    absorber.  Returns z (µm, measured from the source plane), the phasors E_x(z), H_y(z)
    (engineering convention) and the derived ⟨S_z⟩ and p_abs profiles (Meep units)."""
    cell = mp.Vector3(0, 0, length_um)
    z_src = -length_um / 2 + pml_um + source_offset_um
    src = [mp.Source(mp.GaussianSource(frequency=F, fwidth=0.05 * F), component=mp.Ex,
                     center=mp.Vector3(0, 0, z_src))]
    sim = mp.Simulation(cell_size=cell, resolution=resolution, dimensions=1,
                        default_material=meep_lossy_medium(n_re, n_im), sources=src,
                        boundary_layers=[mp.PML(pml_um)])
    z0 = z_src + 0.25                       # start the monitor just past the source plane
    z1 = length_um / 2 - pml_um - 0.25       # stop just before the PML
    span = z1 - z0
    mon = sim.add_dft_fields([mp.Ex, mp.Hy], [F], center=mp.Vector3(0, 0, (z0 + z1) / 2),
                             size=mp.Vector3(0, 0, span))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ex, mp.Vector3(0, 0, z1), 1e-7))
    Ex = np.conj(sim.get_dft_array(mon, mp.Ex, 0))
    Hy = np.conj(sim.get_dft_array(mon, mp.Hy, 0))
    z = np.asarray(sim.get_array_metadata(dft_cell=mon)[2]) - z_src   # exact Yee-grid z, from the source plane, µm
    Sz = 0.5 * np.real(Ex * np.conj(Hy))              # ⟨S_z⟩ = ½ Re{E_x H_y*}
    eps_im = 2 * n_re * n_im
    p_abs = 0.5 * OMEGA * eps_im * np.abs(Ex) ** 2    # notes 25: p_abs = ½ ω ε'' |E|²
    return dict(z=z, Ex=Ex, Hy=Hy, Sz=Sz, p_abs=p_abs, meep_time=sim.meep_time(),
                n_re=n_re, n_im=n_im, resolution=resolution)


def run_tir_interface(neff: float, n1: float = 3.50, n2: float = 1.45, len_z_um: float = 3.0,
                      half_x_um: float = 1.5, margin_um: float = 0.5, pml_um: float = 1.0,
                      resolution: int = 60):
    """2-D FDTD of a flat Si (x<0) / SiO2 (x>0) interface illuminated from the silicon side by a
    plane wave whose along-interface wavenumber is β = n_eff k0 (incidence angle
    θ = asin(n_eff/n1); n_eff > n2 means θ > θ_c, total internal reflection; n_eff < n2 means
    the wave refracts into the silica and carries power across).

    The cell is Bloch-periodic along z with k_z = β, so the whole steady state is exactly
    (something)(x) · e^{-jβz}: any x-dependence is the physics, the z-dependence is imposed.
    Geometry must stay inside the cell: with k_point set, Meep's ensure_periodicity=True wraps
    objects that stick out, which would silently fill the cell with silicon.

    Layout along x (Meep y), from the bottom: PML | margin | monitor (2·half_x) | margin | PML.
    The source line sits in the lower margin (inside silicon, outside both the PML and the
    monitor).  Returns notes-convention arrays on a (z, x) grid: E_y, H_x, H_z, ⟨S_x⟩, ⟨S_z⟩.
    """
    k0 = 2 * np.pi / LAM
    beta = neff * k0
    Ly = 2 * half_x_um + 2 * margin_um + 2 * pml_um
    cell = mp.Vector3(len_z_um, Ly, 0)
    si = mp.Medium(index=n1)
    sio2 = mp.Medium(index=n2)
    # silicon fills the lower half of the cell only (inside the cell: see docstring)
    geom = [mp.Block(size=mp.Vector3(mp.inf, Ly / 2, mp.inf), center=mp.Vector3(0, -Ly / 4), material=si)]
    x_src = -half_x_um - margin_um / 2                # in silicon, between PML and monitor
    src = [mp.Source(mp.GaussianSource(frequency=F, fwidth=0.05 * F), component=mp.Ez,
                     center=mp.Vector3(0, x_src), size=mp.Vector3(len_z_um, 0),
                     amp_func=lambda p: np.exp(1j * beta * p.x))]   # Meep e^{-iωt}: +z wave is e^{+iβz}
    sim = mp.Simulation(cell_size=cell, resolution=resolution, geometry=geom, default_material=sio2,
                        sources=src, boundary_layers=[mp.PML(pml_um, direction=mp.Y)],
                        k_point=mp.Vector3(beta / (2 * np.pi), 0, 0))   # k_point in units of 2π/µm
    mon = sim.add_dft_fields([mp.Ez, mp.Hx, mp.Hy], [F], center=mp.Vector3(0, 0),
                             size=mp.Vector3(len_z_um, 2 * half_x_um))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, mp.Vector3(0, 0.3), 1e-6))
    Ez_m = np.conj(sim.get_dft_array(mon, mp.Ez, 0))   # notes E_y
    Hx_m = np.conj(sim.get_dft_array(mon, mp.Hx, 0))   # notes H_z (along propagation)
    Hy_m = np.conj(sim.get_dft_array(mon, mp.Hy, 0))   # notes H_x (normal to interface)
    eps = sim.get_array(center=mp.Vector3(0, 0), size=mp.Vector3(len_z_um, 2 * half_x_um),
                        component=mp.Dielectric)
    meta = sim.get_array_metadata(dft_cell=mon)          # exact Yee-grid coordinates of the DFT array
    z = np.asarray(meta[0])                               # Meep x = notes z
    x = np.asarray(meta[1])                               # Meep y = notes x
    # Poynting in the notes' axes.  Meep: S = E × H with E = ẑ E_z → S_x^meep = -E_z H_y^meep,
    # S_y^meep = E_z H_x^meep.  Notes: ⟨S_z⟩ = -½Re{E_y H_x*}, ⟨S_x⟩ = ½Re{E_y H_z*}.
    Sz = -0.5 * np.real(Ez_m * np.conj(Hy_m))
    Sx = 0.5 * np.real(Ez_m * np.conj(Hx_m))
    return dict(z=z, x=x, Ey=Ez_m, Hx=Hy_m, Hz=Hx_m, Sx=Sx, Sz=Sz, eps=eps, beta=beta,
                neff=neff, n1=n1, n2=n2, meep_time=sim.meep_time(), resolution=resolution)
