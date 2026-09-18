"""Meep 2-D FDTD of two coupled 220 nm silicon slabs (the evanescent directional coupler).

Geometry (Meep 2-D, cell in x-y, Ez out of plane = the slab TE polarisation of notes §13):
    propagation along +x; slabs stacked along y.
    guide 1 (top, y = +(gap/2 + w/2)) spans the whole cell and carries the source;
    guide 2 (bottom) starts abruptly at x = X_START so that the launched field is exactly the
    single-slab mode, i.e. "all the power in guide 1" (a(0)=a0, b(0)=0 of notes §27).
Must be run with the Meep interpreter (.meep/bin/python).
"""
import numpy as np
import meep as mp

SI = mp.Medium(index=3.50)
SIO2 = mp.Medium(index=1.45)
W = 0.22          # slab thickness [µm]
LAM = 1.31        # vacuum wavelength [µm]; run.py overrides it with set_wavelength(REF.lambda_nm)
F0 = 1 / LAM      # Meep frequency (units c/µm)
SX, SY = 40.0, 5.0
PML = 1.0
X_SRC = -18.0
X_START = -16.0   # where guide 2 begins


def set_wavelength(lam_um):
    """Make every run in this module use vacuum wavelength lam_um (run.py calls it with REF.lambda_nm)."""
    global LAM, F0
    LAM = float(lam_um); F0 = 1 / LAM


def make_sim(gap, resolution=40, both_full_length=False, W2=None):
    """Two slabs: guide 1 (top, thickness W) and guide 2 (bottom, thickness W2, default W = identical guides)."""
    W2 = W if W2 is None else W2
    y1 = +(gap / 2 + W / 2)
    y2 = -(gap / 2 + W2 / 2)
    geom = [mp.Block(size=mp.Vector3(mp.inf, W, mp.inf), center=mp.Vector3(0, y1), material=SI)]
    if both_full_length:
        geom.append(mp.Block(size=mp.Vector3(mp.inf, W2, mp.inf), center=mp.Vector3(0, y2), material=SI))
    else:
        L2 = SX / 2 - X_START + 2.0
        geom.append(mp.Block(size=mp.Vector3(L2, W2, mp.inf), center=mp.Vector3(X_START + L2 / 2, y2), material=SI))
    sim = mp.Simulation(cell_size=mp.Vector3(SX, SY, 0), geometry=geom, default_material=SIO2,
                        resolution=resolution, boundary_layers=[mp.PML(PML)])
    return sim, y1, y2


def eigen_source(y1, src_time):
    # eigenmode of guide 1 alone (guide 2 does not exist yet at x = X_SRC)
    return [mp.EigenModeSource(src_time, center=mp.Vector3(X_SRC, y1), size=mp.Vector3(0, 2.4),
                               eig_band=1, eig_parity=mp.EVEN_Y + mp.ODD_Z, eig_match_freq=True)]


def run_cw(gap, resolution=40, fwidth_frac=0.15, until_after=220, W2=None):
    """Steady-state complex fields at F0 via DFT monitors. Returns dict with x, y, Ez, Hy, Sx, P1, P2."""
    sim, y1, y2 = make_sim(gap, resolution, W2=W2)
    sim.sources = eigen_source(y1, mp.GaussianSource(F0, fwidth=fwidth_frac * F0))
    vol_center = mp.Vector3(0, 0)
    vol_size = mp.Vector3(SX - 2 * PML, SY - 2 * PML)
    dft = sim.add_dft_fields([mp.Ez, mp.Hy], F0, 0, 1, center=vol_center, size=vol_size)
    sim.run(until_after_sources=until_after)
    Ez = sim.get_dft_array(dft, mp.Ez, 0)
    Hy = sim.get_dft_array(dft, mp.Hy, 0)
    x, y, _, _ = sim.get_array_metadata(center=vol_center, size=vol_size)
    x, y = np.array(x), np.array(y)
    # time-averaged Poynting flux along x (notes §25): <S_x> = -1/2 Re(Ez Hy*)  (S = E x H, S_x = E_y H_z - E_z H_y).
    # Power flows +x by construction, so this should already be positive; the guard below is only a safety net
    # and its use is reported (poynting_sign_flipped) so run.py can assert it never fired.
    Sx = -0.5 * np.real(Ez * np.conj(Hy))
    poynting_sign_flipped = bool(Sx.mean() < 0)
    if poynting_sign_flipped:
        Sx = -Sx
    dy = y[1] - y[0]
    top = y > 0.0
    P1 = Sx[:, top].sum(axis=1) * dy    # guide 1 (y > 0 half plane: mid-gap split)
    P2 = Sx[:, ~top].sum(axis=1) * dy   # guide 2
    eps = sim.get_array(center=vol_center, size=vol_size, component=mp.Dielectric)
    sim.reset_meep()
    return dict(gap=gap, x=x, y=y, Ez=Ez, Hy=Hy, Sx=Sx, P1=P1, P2=P2, eps=np.array(eps), y1=y1, y2=y2,
                w1=W, w2=(W if W2 is None else W2), poynting_sign_flipped=poynting_sign_flipped)


def run_pulse(gap, resolution=40, fwidth_frac=0.35, n_frames=180, dt_frame=0.9, downsample=2, W2=None):
    """Time-domain movie of a short pulse hopping between the guides. Returns (t, frames, x, y, eps)."""
    sim, y1, y2 = make_sim(gap, resolution, W2=W2)
    sim.sources = eigen_source(y1, mp.GaussianSource(F0, fwidth=fwidth_frac * F0))
    vol_center = mp.Vector3(0, 0)
    vol_size = mp.Vector3(SX - 2 * PML, SY - 2 * PML)
    frames, times = [], []

    def grab(sim):
        ez = sim.get_array(center=vol_center, size=vol_size, component=mp.Ez)
        frames.append(np.asarray(ez, dtype=np.float32)[::downsample, ::downsample])
        times.append(sim.meep_time())

    sim.run(mp.at_every(dt_frame, grab), until=n_frames * dt_frame)
    x, y, _, _ = sim.get_array_metadata(center=vol_center, size=vol_size)
    eps = np.array(sim.get_array(center=vol_center, size=vol_size, component=mp.Dielectric))
    sim.reset_meep()
    return np.array(times), np.array(frames), np.array(x)[::downsample], np.array(y)[::downsample], eps[::downsample, ::downsample]


def eigen_supermodes(gap, resolution=40, ny=None):
    """Even/odd supermode n_eff and Ez profiles from Meep's eigenmode solver (MPB) on a y cross-section."""
    sim, y1, y2 = make_sim(gap, resolution, both_full_length=True)
    sim.init_sim()
    vol = mp.Volume(center=mp.Vector3(0, 0), size=mp.Vector3(0, SY - 2 * PML))
    out = {}
    ys = np.linspace(-(SY / 2 - PML), SY / 2 - PML, ny or int((SY - 2 * PML) * resolution) + 1)
    for name, par in (("even", mp.EVEN_Y + mp.ODD_Z), ("odd", mp.ODD_Y + mp.ODD_Z)):
        em = sim.get_eigenmode(F0, mp.X, vol, 1, mp.Vector3(2.9 * F0, 0, 0), parity=par)
        prof = np.array([em.amplitude(mp.Vector3(0, yy), mp.Ez) for yy in ys]).real
        prof /= np.max(np.abs(prof))
        out[name] = dict(n_eff=em.k.x / F0, y=ys, F=prof)
    sim.reset_meep()
    return out


def eigen_single(resolution=40):
    """n_eff of one 220 nm slab from Meep's eigenmode solver (for comparison with the analytic slab)."""
    geom = [mp.Block(size=mp.Vector3(mp.inf, W, mp.inf), center=mp.Vector3(0, 0), material=SI)]
    sim = mp.Simulation(cell_size=mp.Vector3(4, SY, 0), geometry=geom, default_material=SIO2,
                        resolution=resolution, boundary_layers=[mp.PML(PML)])
    sim.init_sim()
    em = sim.get_eigenmode(F0, mp.X, mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, SY - 2 * PML)), 1,
                           mp.Vector3(2.9 * F0, 0, 0), parity=mp.EVEN_Y + mp.ODD_Z)
    n = em.k.x / F0
    sim.reset_meep()
    return n
