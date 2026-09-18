"""Meep 2-D FDTD runs for experiment 09: a Gaussian beam on a flat Si/SiO2 interface.

Geometry (Meep cell is x-y, E_z out of plane = the notes' TE field E_y):
    notes' x (normal to the interface)      = Meep x   (silicon at x < 0, silica at x > 0)
    notes' z (along the interface, beta)     = Meep y
    notes' y (direction of the TE E-field)   = Meep z
The beam is launched from a line source at x = SRC_X inside the silicon, focused on the
interface at (0, Y_FOCUS), travelling at angle theta from the interface normal (+x) towards
+y.  Frustrated TIR adds a second silicon half-space at x > gap.

Convention bridge (same as experiment 06): Meep integrates with the physicist's e^{-i omega t};
a Meep DFT phasor A satisfies E(t) = Re{A e^{-i omega t}} = Re{conj(A) e^{+j omega t}}, so every
phasor returned here is conj(Meep DFT) and obeys the notes (+z wave = e^{-j beta z}, evanescent
tail = e^{-gamma x}).  The time-averaged Poynting vector 1/2 Re{E x H*} is the same in both
conventions.  Meep units: lengths in um, frequency = 1/lambda[um], eps0 = mu0 = c = 1.
"""
import sys, pathlib
import numpy as np
import meep as mp

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common import REF

mp.verbosity(0)

LAM = REF.lambda_nm * 1e-3    # um: the ONE wavelength (run.py passes lam=... explicitly to every task)

# cell layout (um)
SX, SY = 10.0, 12.0           # cell size: x normal to the interface, y along it
PML = 1.0
SRC_X = -2.0                  # line source plane inside the silicon (closer = less truncation of the beam)
Y_FOCUS = 0.5                 # beam focus on the interface, shifted so the reflected beam fits
W0 = 2.0                      # beam waist radius (field 1/e) at the focus, um
FWIDTH_REL = 0.2              # pulse bandwidth for the phasor runs, as a fraction of the frequency


def _geometry(n1, n2, interface=True, gap=None):
    """Silicon half-space at x < 0 (optional), silica at x > 0, second silicon at x > gap."""
    si, sio2 = mp.Medium(index=n1), mp.Medium(index=n2)
    if not interface:                       # normalisation run: uniform silicon
        return [], si
    geom = [mp.Block(center=mp.Vector3(-SX / 4, 0), size=mp.Vector3(SX / 2, mp.inf, mp.inf), material=si)]
    if gap is not None:
        w = SX / 2 - gap
        geom.append(mp.Block(center=mp.Vector3(gap + w / 2, 0), size=mp.Vector3(w, mp.inf, mp.inf), material=si))
    return geom, sio2


def _beam_source(theta_deg, src):
    th = np.radians(theta_deg)
    return [mp.GaussianBeamSource(src, center=mp.Vector3(SRC_X, 0), size=mp.Vector3(0, SY - 2 * PML),
                                  beam_x0=mp.Vector3(-SRC_X, Y_FOCUS, 0),
                                  beam_kdir=mp.Vector3(np.cos(th), np.sin(th), 0),
                                  beam_w0=W0, beam_E0=mp.Vector3(0, 0, 1))]


def phasor_run(theta_deg, n1, n2, resolution=40, interface=True, gap=None,
               dft_region=None, flux_x=(-0.3, 0.3), until_after=60.0, lam=LAM):
    """Steady-state phasors at f = 1/lam from a narrow-band pulse + DFT monitors.

    lam: free-space wavelength in um (Meep frequency F = 1/lam); defaults to REF.lambda_nm.
    dft_region: (x_center, x_size) of the DFT box (full interior height); None = no fields.
    flux_x: x positions of flux lines (full interior height); the flux is +x power.
    Returns dict with phasors Ez, Hx, Hy (engineering convention), grid x, y, fluxes.
    """
    F = 1.0 / lam
    geom, default = _geometry(n1, n2, interface, gap)
    src = _beam_source(theta_deg, mp.GaussianSource(F, fwidth=FWIDTH_REL * F))
    sim = mp.Simulation(cell_size=mp.Vector3(SX, SY, 0), geometry=geom, sources=src,
                        default_material=default, resolution=resolution,
                        boundary_layers=[mp.PML(PML)])
    h = SY - 2 * PML
    fluxes = [sim.add_flux(F, 0, 1, mp.FluxRegion(center=mp.Vector3(x, 0), size=mp.Vector3(0, h)))
              for x in flux_x]
    dft = None
    if dft_region is not None:
        xc, xs = dft_region
        dft = sim.add_dft_fields([mp.Ez, mp.Hx, mp.Hy], [F], center=mp.Vector3(xc, 0), size=mp.Vector3(xs, h))
    sim.run(until_after_sources=until_after)
    out = dict(theta_deg=theta_deg, gap=gap, resolution=resolution, lam=lam, meep_time=sim.meep_time(),
               flux_x=list(flux_x), flux=[float(mp.get_fluxes(fl)[0]) for fl in fluxes])
    if dft is not None:
        Ez = np.conj(sim.get_dft_array(dft, mp.Ez, 0))
        Hx = np.conj(sim.get_dft_array(dft, mp.Hx, 0))
        Hy = np.conj(sim.get_dft_array(dft, mp.Hy, 0))
        (x, y, _, _) = sim.get_array_metadata(center=mp.Vector3(xc, 0), size=mp.Vector3(xs, h))
        out.update(Ez=Ez, Hx=Hx, Hy=Hy, x=np.asarray(x), y=np.asarray(y))
    sim.reset_meep()
    return out


def cw_frames(theta_deg, n1, n2, resolution=40, gap=None, t_end=45.0, dt_frame=0.25, view=(-4.0, 4.0), lam=LAM):
    """Continuous-wave run returning real E_z snapshots (frames, x, y) of the interior at wavelength lam (um)."""
    F = 1.0 / lam
    geom, default = _geometry(n1, n2, True, gap)
    src = _beam_source(theta_deg, mp.ContinuousSource(F, width=3.0))
    sim = mp.Simulation(cell_size=mp.Vector3(SX, SY, 0), geometry=geom, sources=src,
                        default_material=default, resolution=resolution,
                        boundary_layers=[mp.PML(PML)])
    xc, xs = 0.5 * (view[0] + view[1]), view[1] - view[0]
    h = SY - 2 * PML
    vol = dict(center=mp.Vector3(xc, 0), size=mp.Vector3(xs, h))
    frames, times = [], []

    def grab(s):
        frames.append(sim.get_array(component=mp.Ez, **vol).astype(np.float32))
        times.append(sim.meep_time())

    sim.run(mp.at_every(dt_frame, grab), until=t_end)
    (x, y, _, _) = sim.get_array_metadata(**vol)
    eps = sim.get_array(component=mp.Dielectric, **vol)
    sim.reset_meep()
    return dict(frames=np.array(frames), times=np.array(times), x=np.asarray(x), y=np.asarray(y), eps=eps, lam=lam)


def phasor_task(kwargs):
    """Picklable wrapper for a process pool (each worker imports meep itself)."""
    return phasor_run(**kwargs)


def cw_task(kwargs):
    r = cw_frames(**kwargs)
    r["frames"] = r["frames"].astype(np.float16)      # halve the pickling cost
    return r


if __name__ == "__main__":       # quick benchmark
    import time
    mp.verbosity(0)
    t0 = time.time()
    r = phasor_run(40.0, 3.5, 1.45, resolution=40, dft_region=(0.0, 8.0))
    print("phasor run 40 deg res 40:", time.time() - t0, "s; flux", r["flux"], r["Ez"].shape)
    t0 = time.time()
    r0 = phasor_run(40.0, 3.5, 1.45, resolution=40, interface=False)
    print("normalisation:", time.time() - t0, "s; flux", r0["flux"])
    t0 = time.time()
    c = cw_frames(40.0, 3.5, 1.45, resolution=40)
    print("cw run:", time.time() - t0, "s; frames", c["frames"].shape, c["frames"].nbytes / 1e6, "MB")
