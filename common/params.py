"""Reference numbers for the Lightmatter capstone (same set the textbook uses).

Every experiment imports from here so that a change in one place propagates.
Sources: [LM] Lightmatter OFC 2025 microring Tx/Rx paper; [P] Padmaraju & Bergman 2013;
[T] Flexcompute thermally tuned ring example; [N] meeting notes; [C] Choi & Stojanovic.
"""
from dataclasses import dataclass

C0 = 299_792_458.0          # m/s
Q_E = 1.602_176_634e-19     # C
M_E = 9.109_383_7e-31       # kg
EPS0 = 8.854_187_8e-12      # F/m
MU0 = 1.256_637_06e-6       # H/m
HBAR = 1.054_571_8e-34      # J s


@dataclass(frozen=True)
class Reference:
    # optical
    lambda_nm: float = 1310.0          # O-band carrier [LM]
    n_si: float = 3.50                 # silicon at 1310 nm (rounded)
    n_sio2: float = 1.45               # silica at 1310 nm (rounded)
    wg_width_um: float = 0.50          # strip width [T]
    wg_height_um: float = 0.22         # strip thickness [T]
    neff: float = 2.5                  # textbook reference value (see experiment 08 for solver values)
    ng: float = 4.2                    # group index (sets the FSR)
    confinement: float = 0.85          # fraction of mode power in silicon
    # ring
    radius_um: float = 6.3
    round_trip_um: float = 39.6
    fsr_thz: float = 1.8               # [LM]
    fsr_nm: float = 10.3
    q_loaded: float = 3500.0           # [LM]
    fwhm_pm: float = 374.0
    a_round_trip: float = 0.945        # field retention per lap (near-critical, doped ring)
    t_coupler: float = 0.945
    kappa2: float = 0.107              # power coupling of the ring-bus coupler
    loss_db_cm_doped: float = 125.0    # equivalent propagation loss of the doped modulator ring
    loss_db_cm_passive: float = 3.0    # undoped strip
    # thermal
    dlambda_dT_pm_per_K: float = 50.0
    dn_si_dT: float = 1.86e-4          # 1/K
    dn_sio2_dT: float = 1.0e-5         # 1/K
    heater_nm_per_mw: float = 0.44     # [LM]
    r_th_K_per_mw: float = 8.8
    tau_th_us: float = 10.0
    tau_slow_us: float = 300.0
    slow_share: float = 0.25
    crosstalk_nearest: float = 0.10
    crosstalk_next: float = 0.03
    ring_pitch_um: float = 15.0        # [N]
    ambient_min_c: float = 10.0        # [N]
    ambient_max_c: float = 125.0       # [N]
    # modulator / link
    channel_spacing_ghz: float = 200.0
    n_channels: int = 8
    mod_eff_pm_per_v: float = 50.0     # [LM]
    swing_vpp: float = 1.3             # [LM]
    p_in_dbm: float = 4.0              # [LM]
    responsivity_a_per_w: float = 0.9  # [LM]
    baud: float = 53.125e9
    t_min: float = 0.016               # on-resonance through transmission (18 dB)
    delta_opt_pm: float = 108.0        # max-OMA bias, FWHM/(2 sqrt 3)


REF = Reference()


def k0_per_um(lambda_nm: float = REF.lambda_nm) -> float:
    """Vacuum wavenumber in rad/um."""
    import math
    return 2 * math.pi / (lambda_nm * 1e-3)


def omega_rad_s(lambda_nm: float = REF.lambda_nm) -> float:
    import math
    return 2 * math.pi * C0 / (lambda_nm * 1e-9)


def freq_thz(lambda_nm: float = REF.lambda_nm) -> float:
    return C0 / (lambda_nm * 1e-9) / 1e12
