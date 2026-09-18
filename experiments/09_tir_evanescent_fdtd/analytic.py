"""Analytic expectations for experiment 09 (notes sections 4, 17, 24, 25, 27).

Coordinates follow docs/NOTES.md: x is normal to the interface (silicon at x < 0,
silica at x > 0), z runs along the interface, the TE electric field is E_y (out of
the plane).  A plane wave in silicon with angle of incidence theta (from the normal)
has the shared tangential wavevector beta = n1 k0 sin(theta); in silica
k_x^2 = n2^2 k0^2 - beta^2, which is negative beyond the critical angle, so
k_x = -j*gamma with gamma = sqrt(beta^2 - n2^2 k0^2)  (notes 17, 24).

Everything is s-polarisation (TE, E out of plane), which is what a 2-D Meep run
with Ez computes.  Units: lengths in um, wavenumbers in rad/um.
"""
import numpy as np


def kx_medium(n, k0, beta):
    """Normal wavenumber in a medium of index n for tangential wavenumber beta.

    Returns a complex array: real (propagating) when beta < n k0, and -j*gamma
    (decaying towards +x, e^{-j kx x} = e^{-gamma x}) when beta > n k0.
    """
    arg = (n * k0) ** 2 - np.asarray(beta, dtype=float) ** 2
    kx = np.where(arg >= 0, np.sqrt(np.abs(arg)), -1j * np.sqrt(np.abs(arg)))
    return kx.astype(complex)


def gamma_evanescent(n1, n2, k0, theta_deg):
    """Decay constant gamma (1/um) and decay length 1/gamma (um) beyond critical."""
    beta = n1 * k0 * np.sin(np.radians(theta_deg))
    g = np.sqrt(np.maximum(beta**2 - (n2 * k0) ** 2, 0.0))
    with np.errstate(divide="ignore"):
        return g, np.where(g > 0, 1.0 / g, np.inf)


def fresnel_s(n1, n2, k0, beta):
    """Single interface, s-polarisation: r, t (field), T (power fraction)."""
    kx1 = kx_medium(n1, k0, beta)
    kx2 = kx_medium(n2, k0, beta)
    r = (kx1 - kx2) / (kx1 + kx2)
    t = 2 * kx1 / (kx1 + kx2)
    T = np.real(kx2) / np.real(kx1) * np.abs(t) ** 2
    return r, t, T


def ftir_s(n1, n2, k0, beta, gap):
    """Si | silica gap | Si, s-polarisation: field transmission and power fraction.

    Transfer through the gap:  t = t12 t23 e^{-j kx2 d} / (1 + r12 r23 e^{-2j kx2 d}),
    with r23 = -r12 and t23 = 2 kx2/(kx2 + kx1).  Beyond critical kx2 = -j gamma and
    the exponentials become e^{-gamma d}: the tunnelling factor.
    """
    kx1 = kx_medium(n1, k0, beta)
    kx2 = kx_medium(n2, k0, beta)
    r12 = (kx1 - kx2) / (kx1 + kx2)
    t12 = 2 * kx1 / (kx1 + kx2)
    r23 = -r12
    t23 = 2 * kx2 / (kx2 + kx1)
    ph = np.exp(-1j * kx2 * gap)
    t = t12 * t23 * ph / (1 + r12 * r23 * ph**2)
    return t, np.abs(t) ** 2


def ftir_closed_form(n1, n2, k0, theta_deg, gap):
    """Textbook closed form beyond critical: T = 1 / (1 + (k1^2+g^2)^2/(4 k1^2 g^2) sinh^2(g d))."""
    th = np.radians(theta_deg)
    k1 = n1 * k0 * np.cos(th)
    g, _ = gamma_evanescent(n1, n2, k0, theta_deg)
    return 1.0 / (1.0 + (k1**2 + g**2) ** 2 / (4 * k1**2 * g**2) * np.sinh(g * gap) ** 2)


class GaussianBeamSpectrum:
    """Plane-wave (angular) spectrum of the 2-D Gaussian beam used in the FDTD run.

    The beam is focused on the interface at the origin with waist radius w0 (field 1/e)
    and central angle theta.  Along the interface line x = 0 the incident field is
    E(z) = exp(-(z cos(theta))^2 / w0^2) * exp(-j beta0 z), whose Fourier transform is a
    Gaussian in beta centred on beta0 = n1 k0 sin(theta) with 1/e half-width
    2 cos(theta)/w0.  Each plane-wave component carries its own kx, gamma, r and t, so
    everything the finite beam does (partial transmission near the critical angle, a
    decay that is not a single exponential, the Goos-Haenchen loop of power) follows
    from summing the components.
    """

    def __init__(self, n1, n2, k0, theta_deg, w0, nbeta=4001):
        self.n1, self.n2, self.k0, self.w0 = n1, n2, k0, w0
        self.theta = np.radians(theta_deg)
        self.beta0 = n1 * k0 * np.sin(self.theta)
        half = 2 * np.cos(self.theta) / w0
        self.beta = np.linspace(self.beta0 - 6 * half, self.beta0 + 6 * half, nbeta)
        # keep only propagating components in silicon
        self.beta = self.beta[np.abs(self.beta) < n1 * k0 * 0.999]
        self.A = np.exp(-(((self.beta - self.beta0) / half) ** 2))
        self.kx1 = kx_medium(n1, k0, self.beta)
        self.kx2 = kx_medium(n2, k0, self.beta)
        self.r, self.t, self.T = fresnel_s(n1, n2, k0, self.beta)
        self.weight = np.real(self.kx1) * self.A**2  # power per component crossing x = 0
        self.theta_deg = theta_deg

    def transmission_beam(self, gap=None):
        """Power-weighted average of the plane-wave transmission over the beam spectrum."""
        if gap is None:
            T = self.T
        else:
            _, T = ftir_s(self.n1, self.n2, self.k0, self.beta, gap)
        return float(np.sum(self.weight * T) / np.sum(self.weight))

    def transmitted_field(self, x, z):
        """Complex E_y on the silica side (x >= 0) for the single interface.

        E(x, z) = sum_beta A(beta) t(beta) e^{-j kx2(beta) x} e^{-j beta z}.
        Returns an array of shape (len(x), len(z)).
        """
        x = np.atleast_1d(x)[:, None, None]
        z = np.atleast_1d(z)[None, :, None]
        b = self.beta[None, None, :]
        ph = np.exp(-1j * self.kx2[None, None, :] * x) * np.exp(-1j * b * z)
        return np.sum(self.A[None, None, :] * self.t[None, None, :] * ph, axis=-1)

    def decay_at_z(self, x, z=0.0):
        """|E_y(x, z)| normalised to the interface value: the beam's actual evanescent decay."""
        e = self.transmitted_field(x, [z])[:, 0]
        return np.abs(e) / np.abs(self.transmitted_field([0.0], [z])[0, 0])


def fit_decay_length(x, amp, x_min, x_max):
    """Fit ln|E| = a - x/L over x_min < x < x_max; return L (same units as x) and the fit."""
    m = (x > x_min) & (x < x_max) & (amp > 0)
    p = np.polyfit(x[m], np.log(amp[m]), 1)
    return -1.0 / p[0], p


def poynting_2d(Ez, Hx, Hy):
    """Time-averaged Poynting vector (Meep coordinates) from complex phasors.

    S = 1/2 Re(E x H*) with E = z_hat Ez, H = (Hx, Hy, 0):
    S_x = -1/2 Re(Ez Hy*),  S_y = 1/2 Re(Ez Hx*).
    In the notes' coordinates Meep-x is the normal x and Meep-y is the along-interface z.
    """
    Sx = -0.5 * np.real(Ez * np.conj(Hy))
    Sy = 0.5 * np.real(Ez * np.conj(Hx))
    return Sx, Sy
