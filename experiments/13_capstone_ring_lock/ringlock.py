"""Models for 13_capstone_ring_lock.

Single-ring thermal lock, written so that run.py and explore.ipynb share one implementation:

    heater power P_h [mW] --(R_th, two thermal poles)--> ring temperature rise ΔT [K]
    ΔT --(dλ_r/dT = 50 pm/K)--> resonance shift  -->  detuning δ = λ_L − λ_r [pm]
    δ --(through-port Lorentzian, T_min, FWHM)--> transmission T(δ)
    T --(P_in · tap · responsivity)--> photocurrent I [µA]  --> PI controller --> P_h

Conventions (docs/NOTES.md): δ = λ_L − λ_r, positive when the laser sits on the red side of the
resonance. Heating red-shifts λ_r, so heating DEcreases δ. With the laser on the red slope the
photocurrent falls as the ring heats: dI/dP_h < 0. The controller error is defined as
e = I_meas − I_set, so a positive error ("too much light, ring too cold, notch too far blue")
means "heat more", and the gains K_p, K_i are positive numbers.

Units inside this module: time s, power mW, temperature K, wavelength pm, current µA.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import control as ct
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common import REF

# --------------------------------------------------------------------------------------------
# Thermal plant: heater power -> ring temperature rise (two-pole, R_th DC gain)
# --------------------------------------------------------------------------------------------
@dataclass
class Plant:
    r_th: float = REF.r_th_K_per_mw            # K/mW
    tau_fast: float = REF.tau_th_us * 1e-6     # s   (75 % of the response)
    tau_slow: float = REF.tau_slow_us * 1e-6   # s   (25 % slow tail: substrate / oxide)
    slow_share: float = REF.slow_share
    dlam_dT: float = REF.dlambda_dT_pm_per_K   # pm/K

    @property
    def heater_pm_per_mw(self) -> float:
        """Heater tuning efficiency dλ_r/dP_h = R_th · dλ_r/dT (pm/mW). REF says 0.44 nm/mW."""
        return self.r_th * self.dlam_dT

    def thermal_tf(self, normalised: bool = False) -> ct.TransferFunction:
        """G_th(s) = R_th [ (1−s_sh)/(1+sτ_f) + s_sh/(1+sτ_s) ]  (K/mW), or unity-DC-gain version."""
        s = ct.tf("s")
        g = (1 - self.slow_share) / (1 + s * self.tau_fast) + self.slow_share / (1 + s * self.tau_slow)
        return g if normalised else self.r_th * g

    def thermal_step(self, t: np.ndarray, p_mw: float = 1.0):
        """Analytic step response ΔT(t) and its fast / slow pieces (K)."""
        fast = self.r_th * p_mw * (1 - self.slow_share) * (1 - np.exp(-t / self.tau_fast))
        slow = self.r_th * p_mw * self.slow_share * (1 - np.exp(-t / self.tau_slow))
        return fast + slow, fast, slow


# --------------------------------------------------------------------------------------------
# Sensor: detuning -> through-port transmission -> tapped photocurrent
# --------------------------------------------------------------------------------------------
@dataclass
class Sensor:
    fwhm_pm: float = REF.fwhm_pm
    t_min: float = REF.t_min
    p_in_mw: float = 10 ** (REF.p_in_dbm / 10)       # 4 dBm = 2.512 mW (the brief rounds to 2.5)
    tap: float = 0.05
    responsivity: float = REF.responsivity_a_per_w   # A/W
    mod_swing_pm: float = REF.mod_eff_pm_per_v * REF.swing_vpp   # 50 pm/V × 1.3 Vpp = 65 pm

    @property
    def i_fs_ua(self) -> float:
        """Full-scale photocurrent (T = 1) in µA: R · P_in · tap."""
        return self.responsivity * self.p_in_mw * self.tap * 1e3

    @property
    def delta_opt_pm(self) -> float:
        """Bias for maximum small-signal slope of a Lorentzian: FWHM/(2√3)."""
        return self.fwhm_pm / (2 * np.sqrt(3))

    def T(self, delta_pm):
        """Through-port power transmission, Lorentzian notch: 1 − (1−T_min)/(1 + (2δ/FWHM)²)."""
        u = 2 * np.asarray(delta_pm, float) / self.fwhm_pm
        return 1 - (1 - self.t_min) / (1 + u ** 2)

    def dT(self, delta_pm):
        """dT/dδ (1/pm), analytic."""
        d = np.asarray(delta_pm, float)
        u = 2 * d / self.fwhm_pm
        return (1 - self.t_min) * (8 * d / self.fwhm_pm ** 2) / (1 + u ** 2) ** 2

    def dT_max_analytic(self) -> float:
        """max dT/dδ = (1−T_min)·3√3/(4·FWHM), reached at δ = FWHM/(2√3)."""
        return (1 - self.t_min) * 3 * np.sqrt(3) / (4 * self.fwhm_pm)

    def I(self, delta_pm):
        """Photocurrent (µA) = I_fs · T(δ)."""
        return self.i_fs_ua * self.T(delta_pm)

    def dI(self, delta_pm):
        """Sensor gain dI/dδ (µA/pm)."""
        return self.i_fs_ua * self.dT(delta_pm)

    def I_modulated_mean(self, delta_pm, swing_pm=None):
        """Mean photocurrent with NRZ data on: average of the two levels λ_r ± swing/2."""
        sw = self.mod_swing_pm if swing_pm is None else swing_pm
        return 0.5 * (self.I(delta_pm + sw / 2) + self.I(delta_pm - sw / 2))

    def oma_mw(self, delta_bias_pm, swing_pm=None):
        """Optical modulation amplitude at the bus output (mW) for data levels λ_r ± swing/2."""
        sw = self.mod_swing_pm if swing_pm is None else swing_pm
        d = np.asarray(delta_bias_pm, float)
        return self.p_in_mw * np.abs(self.T(d + sw / 2) - self.T(d - sw / 2))

    def er_db(self, delta_bias_pm, swing_pm=None):
        sw = self.mod_swing_pm if swing_pm is None else swing_pm
        d = np.asarray(delta_bias_pm, float)
        hi = np.maximum(self.T(d + sw / 2), self.T(d - sw / 2)); lo = np.minimum(self.T(d + sw / 2), self.T(d - sw / 2))
        return 10 * np.log10(hi / lo)


# --------------------------------------------------------------------------------------------
# Discrete PI controller with saturation, anti-windup, optional acquisition sweep and DAC bits
# --------------------------------------------------------------------------------------------
@dataclass
class PIController:
    kp: float                     # mW/µA
    ki: float                     # mW/(µA·s)
    i_set: float                  # µA setpoint (photocurrent at δ_opt)
    ts: float = 10e-6             # sample period (s); measurement = mean over the previous period
    p_min: float = 0.0            # heater floor (mW): a heater can only heat
    p_max: float = 20.0           # heater ceiling (mW), assumed
    anti_windup: bool = True
    acquire: bool = False         # start in sweep mode (heater ramps up until the notch is found)
    sweep_rate: float = 4000.0    # mW/s while sweeping (4 mW/ms -> 1.76 nm/ms)
    i_capture: float = np.inf     # switch sweep -> PI once I_meas < i_capture (µA)
    dac_bits: int | None = None   # heater DAC resolution over [0, p_max]; None = ideal
    p_init: float = 0.0           # initial heater power (mW); the integrator starts here (bumpless)
    # state
    x_i: float = field(init=False)
    p_out: float = field(init=False)
    mode: str = field(init=False)
    def __post_init__(self):
        self.x_i = float(self.p_init); self.p_out = float(self.p_init)
        self.mode = "sweep" if self.acquire else "pi"

    def update(self, i_meas: float) -> float:
        if self.mode == "sweep":
            self.p_out = min(self.p_out + self.sweep_rate * self.ts, self.p_max)
            if i_meas < self.i_capture:
                self.mode = "pi"; self.x_i = self.p_out      # bumpless transfer
            return self._dac(self.p_out)
        e = i_meas - self.i_set                                 # µA; >0 -> ring too cold -> heat
        self.x_i += self.ki * self.ts * e                       # backward-Euler integrator (mW)
        u = self.kp * e + self.x_i
        u_sat = min(max(u, self.p_min), self.p_max)
        if self.anti_windup and u != u_sat:
            self.x_i = u_sat - self.kp * e                      # clamp: integrator stops at the rail
        self.p_out = u_sat
        return self._dac(self.p_out)

    def _dac(self, p):
        if self.dac_bits is None:
            return p
        lsb = self.p_max / (2 ** self.dac_bits - 1)
        return round(p / lsb) * lsb


# --------------------------------------------------------------------------------------------
# Time-domain simulator for N rings sharing a substrate (thermal crosstalk matrix K)
# --------------------------------------------------------------------------------------------
def crosstalk_matrix(n: int, nearest: float = REF.crosstalk_nearest, next_: float = REF.crosstalk_next) -> np.ndarray:
    K = np.eye(n)
    for i in range(n):
        for j in range(n):
            if abs(i - j) == 1: K[i, j] = nearest
            elif abs(i - j) == 2: K[i, j] = next_
    return K


def nominal_heater_mw(plant: Plant, t_amb_c: float, t_amb_hot_c: float = REF.ambient_max_c, p_margin_mw: float = 0.5) -> float:
    """Heater power that puts δ = δ_opt at ambient t_amb_c, given the design rule
    'at the hottest ambient (125 C) the heater still has p_margin_mw of headroom above zero'."""
    return p_margin_mw + (t_amb_hot_c - t_amb_c) / plant.r_th


@dataclass
class SimResult:
    t: np.ndarray            # s
    delta: np.ndarray        # (n_t, n_rings) pm
    i: np.ndarray            # µA (instantaneous)
    p: np.ndarray            # mW (heater, held per tick)
    dT: np.ndarray           # K ring temperature rise above the reference ambient
    t_amb: np.ndarray        # K ambient deviation (n_t, n_rings)
    mode: np.ndarray         # (n_ticks, n_rings) 0 = sweep, 1 = pi
    t_tick: np.ndarray       # s, controller ticks
    i_meas: np.ndarray       # (n_ticks, n_rings) averaged photocurrent used by the controller


def simulate(plant: Plant, sensor: Sensor, ctrls: list[PIController], t_end: float,
             ambient_fn=None, K: np.ndarray | None = None, dt: float = 1e-6,
             delta_cold_pm: float | np.ndarray | None = None, p_nominal_mw: float | None = None,
             i_noise_ua: float = 0.0, seed: int = 0, lock_enabled=None) -> SimResult:
    """Integrate the N-ring plant with exact first-order updates for the two thermal states.

    delta_cold_pm: detuning with zero heater power at the reference ambient. Default: δ_opt + η·p_nominal,
    so that P_h = p_nominal at zero ambient deviation gives δ = δ_opt for every ring.
    ambient_fn(t) -> array (n_rings,) of ambient deviation from the reference (K); it is passed through
    the same two-pole thermal filter as the heater (the ring cannot jump in temperature).
    lock_enabled: list of bools; a ring with lock disabled holds its p_init (open loop).
    """
    n = len(ctrls)
    K = np.eye(n) if K is None else np.asarray(K, float)
    ts = ctrls[0].ts
    n_sub = int(round(ts / dt)); dt = ts / n_sub
    n_ticks = int(np.ceil(t_end / ts)); n_t = n_ticks * n_sub
    rng = np.random.default_rng(seed)
    if p_nominal_mw is None:
        p_nominal_mw = nominal_heater_mw(plant, 60.0)
    eta = plant.heater_pm_per_mw
    delta_cold = np.full(n, sensor.delta_opt_pm + eta * p_nominal_mw) if delta_cold_pm is None else np.broadcast_to(np.asarray(delta_cold_pm, float), (n,)).copy()
    lock_enabled = [True] * n if lock_enabled is None else list(lock_enabled)
    if ambient_fn is None:
        ambient_fn = lambda t: np.zeros(n)
    sh = plant.slow_share
    ef, es = np.exp(-dt / plant.tau_fast), np.exp(-dt / plant.tau_slow)
    # initial thermal states: steady state for the initial heater powers and ambient at t=0
    P = np.array([c.p_out for c in ctrls])
    u0 = plant.r_th * (K @ P) + np.asarray(ambient_fn(0.0), float)
    xf, xs = (1 - sh) * u0, sh * u0
    T_ = np.empty((n_t, n)); D_ = np.empty((n_t, n)); I_ = np.empty((n_t, n)); P_ = np.empty((n_t, n)); A_ = np.empty((n_t, n))
    MODE = np.empty((n_ticks, n)); IM = np.empty((n_ticks, n)); TT = np.arange(n_ticks) * ts
    dT0 = xf + xs
    i_avg = sensor.I(delta_cold - plant.dlam_dT * dT0)
    row = 0
    for k in range(n_ticks):
        meas = i_avg + (rng.normal(0.0, i_noise_ua, n) if i_noise_ua > 0 else 0.0)
        for j, c in enumerate(ctrls):
            if lock_enabled[j]:
                P[j] = c.update(float(meas[j]))
            MODE[k, j] = 0.0 if c.mode == "sweep" else 1.0
        IM[k] = meas
        i_sum = np.zeros(n)
        for m in range(n_sub):
            t = (k * n_sub + m + 1) * dt
            amb = np.asarray(ambient_fn(t), float)
            u = plant.r_th * (K @ P) + amb
            xf = xf * ef + (1 - sh) * u * (1 - ef)
            xs = xs * es + sh * u * (1 - es)
            dT = xf + xs
            delta = delta_cold - plant.dlam_dT * dT
            i_now = sensor.I(delta)
            i_sum += i_now
            T_[row] = dT; D_[row] = delta; I_[row] = i_now; P_[row] = P; A_[row] = amb
            row += 1
        i_avg = i_sum / n_sub
    t = np.arange(1, n_t + 1) * dt
    return SimResult(t, D_, I_, P_, T_, A_, MODE, TT, IM)


# --------------------------------------------------------------------------------------------
# Linear design with python-control
# --------------------------------------------------------------------------------------------
def loop_gain_static(plant: Plant, sensor: Sensor, delta_bias_pm: float | None = None) -> float:
    """K_I·K_λ·R_th: photocurrent change per mW of heater at the bias point (µA/mW), sign dropped."""
    d = sensor.delta_opt_pm if delta_bias_pm is None else delta_bias_pm
    return float(sensor.dI(d) * plant.dlam_dT * plant.r_th)


def pi_tf(kp: float, ki: float) -> ct.TransferFunction:
    s = ct.tf("s")
    return kp + ki / s


def delay_tf(t_d: float, order: int = 3) -> ct.TransferFunction:
    num, den = ct.pade(t_d, order)
    return ct.tf(num, den)


def design_pi(plant: Plant, sensor: Sensor, t_d: float = 10e-6, wc: float | None = None,
              tau_i: float | None = None, delta_bias_pm: float | None = None) -> dict:
    """PI by pole cancellation: zero at 1/tau_i (default: the fast thermal pole), K_p from |L(jω_c)| = 1
    with ω_c = 1/(2 T_d) by default. The delay has unit magnitude so it does not enter the gain equation."""
    wc = 1 / (2 * t_d) if wc is None else wc
    tau_i = plant.tau_fast if tau_i is None else tau_i
    d = sensor.delta_opt_pm if delta_bias_pm is None else delta_bias_pm
    K_I = float(sensor.dI(d))                          # µA/pm
    Kstat = K_I * plant.dlam_dT                        # µA/K
    Pj = complex(plant.thermal_tf()(1j * wc))          # K/mW at ω_c
    Cj_over_kp = 1 + 1 / (1j * wc * tau_i)
    kp = 1 / abs(Cj_over_kp * Pj * Kstat)              # mW/µA
    ki = kp / tau_i
    return dict(kp=kp, ki=ki, tau_i=tau_i, wc=wc, t_d=t_d, K_I=K_I, K_lambda=plant.dlam_dT, Kstat=Kstat,
                Kv=ki * Kstat * plant.r_th)            # velocity constant (1/s): lim s·L(s)


def loop_tf(plant: Plant, sensor: Sensor, kp: float, ki: float, t_d: float, pade_order: int = 3,
            delta_bias_pm: float | None = None):
    """Returns (L with Pade delay, L without delay, C, P_K) where P_K = thermal plant × K_λ × K_I."""
    d = sensor.delta_opt_pm if delta_bias_pm is None else delta_bias_pm
    C = pi_tf(kp, ki)
    PK = plant.thermal_tf() * (plant.dlam_dT * float(sensor.dI(d)))
    L0 = C * PK
    return L0 * delay_tf(t_d, pade_order), L0, C, PK


def exact_margins(L0: ct.TransferFunction, t_d: float, w: np.ndarray) -> dict:
    """Gain/phase margins using the exact e^{-jωT_d} on a frequency grid (cross-check of the Pade result)."""
    r = ct.frequency_response(L0, w)
    H = r.magnitude * np.exp(1j * r.phase) * np.exp(-1j * w * t_d)
    mag, ph = np.abs(H), np.unwrap(np.angle(H))
    # gain crossover: first downward crossing of |L| = 1, linear interpolation in log-frequency
    ldb = 20 * np.log10(mag)
    idx = np.where((ldb[:-1] >= 0) & (ldb[1:] < 0))[0]
    if len(idx):
        i1 = idx[0]; fr = ldb[i1] / (ldb[i1] - ldb[i1 + 1])
        wc = float(np.exp(np.log(w[i1]) + fr * (np.log(w[i1 + 1]) - np.log(w[i1]))))
        ph_c = float(np.interp(wc, w, ph)); pm = 180 + np.degrees(ph_c)
    else:
        wc, pm = np.nan, np.nan
    # phase crossover (-180°)
    idx2 = np.where(np.diff(np.sign(ph + np.pi)) != 0)[0]
    if len(idx2):
        i2 = idx2[0]
        w180 = float(np.interp(-np.pi, ph[i2:i2 + 2][::-1], w[i2:i2 + 2][::-1]))
        gm = 1 / float(np.interp(w180, w, mag))
    else:
        w180, gm = np.nan, np.inf
    return dict(wc=wc, pm_deg=pm, w180=w180, gm=gm, gm_db=20 * np.log10(gm) if np.isfinite(gm) else np.inf,
                mag=mag, phase_deg=np.degrees(ph), w=w)
