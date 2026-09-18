#!/usr/bin/env python
"""Environment check for photonics-simulations.

Run with the main interpreter:

    /Users/kennyg/photonics-simulations/.venv/bin/python /Users/kennyg/photonics-simulations/env/check_env.py

For every tool the suite uses it prints the version and the location on this machine,
runs a short smoke test (import plus one trivial call for Python libraries; a version
query for command-line programs; a subprocess "import meep" for the separate Meep
environment) and prints one OK / FAIL line per tool. Exit status is 1 if anything FAILs.
Nothing is written to disk except a temporary directory that is removed at the end.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
MEEP_PY = os.path.join(ROOT, ".meep", "bin", "python")
JUPYTER = os.path.join(ROOT, ".venv", "bin", "jupyter")
MANIM = os.path.join(ROOT, ".venv", "bin", "manim")
NGSPICE = "/opt/homebrew/bin/ngspice"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
MICROMAMBA = "/opt/homebrew/bin/micromamba"
UV = "/opt/homebrew/bin/uv"
BREW = "/opt/homebrew/bin/brew"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

RESULTS: list[tuple[str, bool, str, float]] = []


def report(name: str, ok: bool, detail: str, dt: float) -> None:
    RESULTS.append((name, ok, detail, dt))
    print(f"{'OK  ' if ok else 'FAIL'}  {name:<34} {detail}  ({dt:.2f} s)", flush=True)


def check(name: str, fn):
    """Run fn(); it returns a detail string. Any exception is a FAIL."""
    t0 = time.time()
    try:
        detail = fn()
        report(name, True, detail, time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        report(name, False, f"{type(exc).__name__}: {exc} [{tb}]", time.time() - t0)


def version_of(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "?"


def where(modname: str) -> str:
    mod = importlib.import_module(modname)
    path = getattr(mod, "__file__", None) or "(namespace)"
    return os.path.relpath(path, ROOT) if path.startswith(ROOT) else path


def run_cmd(args: list[str], timeout: float = 60.0) -> str:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout + p.stderr).strip()
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {out[-300:]}")
    return out


# ---------------------------------------------------------------------------
# interpreters and package managers
# ---------------------------------------------------------------------------
def t_python():
    if os.path.realpath(sys.executable) != os.path.realpath(VENV_PY):
        raise RuntimeError(f"run this with {VENV_PY}, not {sys.executable}")
    return f"Python {sys.version.split()[0]} at {os.path.relpath(sys.executable, ROOT)}"


def t_uv():
    return run_cmd([UV, "--version"]).splitlines()[0] + f" at {UV}"


def t_brew():
    return run_cmd([BREW, "--version"]).splitlines()[0] + f" at {BREW}"


def t_micromamba():
    v = run_cmd([MICROMAMBA, "--version"]).splitlines()[0]
    root = os.path.join(ROOT, ".mamba")
    if not os.path.isdir(root):
        raise RuntimeError(".mamba (MAMBA_ROOT_PREFIX) is missing")
    return f"micromamba {v} at {MICROMAMBA}; root prefix .mamba/"


def t_meep():
    if not os.path.exists(MEEP_PY):
        raise RuntimeError(f"{MEEP_PY} does not exist")
    code = (
        "import meep as mp, numpy, scipy, matplotlib, h5py, imageio, sys;"
        "m = mp.Medium(index=3.5); assert abs(m.epsilon_diag.x - 12.25) < 1e-9;"
        "assert hasattr(mp, 'Harminv') and hasattr(mp, 'EigenModeSource');"
        "print(mp.__version__, numpy.__version__, scipy.__version__, matplotlib.__version__, sys.version.split()[0])"
    )
    out = run_cmd([MEEP_PY, "-c", code], timeout=120)
    line = [l for l in out.splitlines() if l and not l.startswith("Elapsed")][-1]
    mv, nv, sv, mplv, pyv = line.split()
    libs = [f for f in os.listdir(os.path.join(ROOT, ".meep", "lib")) if f in ("libmeep.dylib", "libmpb.dylib", "libharminv.dylib")]
    if len(libs) != 3:
        raise RuntimeError(f"expected libmeep/libmpb/libharminv in .meep/lib, found {libs}")
    return f"meep {mv} (Python {pyv}, numpy {nv}, scipy {sv}, matplotlib {mplv}) at .meep/bin/python; libmeep, libmpb, libharminv present"


def t_harminv_binary():
    exe = os.path.join(ROOT, ".meep", "bin", "harminv")
    if not os.path.exists(exe):
        raise RuntimeError("harminv executable missing from .meep/bin")
    out = run_cmd([exe, "-V"])
    return out.splitlines()[0] + " at .meep/bin/harminv"


# ---------------------------------------------------------------------------
# command-line programs
# ---------------------------------------------------------------------------
def t_ngspice():
    out = run_cmd([NGSPICE, "--version"])
    line = next(l for l in out.splitlines() if "ngspice" in l.lower())
    return line.strip("* ").strip() + f" at {NGSPICE}"


def t_ngspice_batch():
    """A one-resistor DC operating point in batch mode, in a temp dir."""
    d = tempfile.mkdtemp(prefix="check_env_")
    try:
        cir = os.path.join(d, "t.cir")
        with open(cir, "w") as f:
            f.write("check\nV1 1 0 DC 2\nR1 1 0 1k\n.control\nop\nprint v(1)\nquit\n.endc\n.end\n")
        out = run_cmd([NGSPICE, "-b", cir])
        if "2.000000e+00" not in out:
            raise RuntimeError(f"unexpected output: {out[-200:]}")
        return "batch run of a 1-resistor netlist gives v(1) = 2 V"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_ffmpeg():
    return run_cmd([FFMPEG, "-version"]).splitlines()[0].split("Copyright")[0].strip() + f" at {FFMPEG}"


def t_ffprobe():
    return run_cmd([FFPROBE, "-version"]).splitlines()[0].split("Copyright")[0].strip() + f" at {FFPROBE}"


def t_jupyter():
    out = run_cmd([JUPYTER, "--version"])
    lab = next((l.split(":")[1].strip() for l in out.splitlines() if l.startswith("jupyterlab")), "?")
    ks = run_cmd([JUPYTER, "kernelspec", "list"])
    if "photonics-sims" not in ks:
        raise RuntimeError("kernel 'photonics-sims' is not installed (jupyter kernelspec list)")
    return f"jupyterlab {lab} at .venv/bin/jupyter; kernel photonics-sims installed"


def t_manim_cli():
    out = run_cmd([MANIM, "--version"], timeout=120)
    return out.strip().splitlines()[0] + " at .venv/bin/manim"


def t_no_latex():
    found = [t for t in ("latex", "pdflatex", "xelatex") if shutil.which(t)]
    return "no LaTeX on this machine (expected: Manim uses Text(), matplotlib uses mathtext)" if not found else f"LaTeX found: {found} (not required)"


# ---------------------------------------------------------------------------
# Python libraries in .venv
# ---------------------------------------------------------------------------
def t_numpy():
    import numpy as np
    x = np.fft.ifft(np.fft.fft(np.arange(8.0)))
    assert np.allclose(x.real, np.arange(8.0))
    return f"numpy {np.__version__} at {where('numpy')}; fft round trip"


def t_scipy():
    import scipy
    from scipy.optimize import brentq
    from scipy.signal import find_peaks
    r = brentq(lambda x: x * x - 2, 0, 2)
    assert abs(r - 2 ** 0.5) < 1e-12
    assert len(find_peaks([0, 1, 0, 1, 0])[0]) == 2
    return f"scipy {scipy.__version__} at {where('scipy')}; brentq sqrt(2), find_peaks"


def t_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"$E(z,t)=E_0\cos(\omega t-\beta z)$")  # mathtext, no LaTeX needed
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG
    assert FFMpegWriter.isAvailable(), "FFMpegWriter cannot find ffmpeg"
    return f"matplotlib {matplotlib.__version__} at {where('matplotlib')}; Agg figure with mathtext, FFMpegWriter available"


def t_sympy():
    import sympy as sp
    lam = sp.symbols("lambda", positive=True)
    n = sp.Function("n")(lam)
    ng = n - lam * sp.diff(n, lam)
    assert sp.simplify(ng.subs(n, lam ** 2) - (-lam ** 2)) == 0
    return f"sympy {sp.__version__} at {where('sympy')}; symbolic n_g = n - lambda dn/dlambda"


def t_control():
    import control
    G = control.tf([1], [1e-5, 1])
    L = control.series(control.tf([1], [1, 0]), G)
    gm, pm, wg, wp = control.margin(L)
    assert pm > 0
    return f"python-control {control.__version__} at {where('control')}; tf, series, margin (PM {pm:.1f} deg)"


def t_skrf():
    import skrf
    f = skrf.Frequency(1, 2, 3, "GHz")
    assert len(f.f) == 3
    return f"scikit-rf {skrf.__version__} at {where('skrf')} (installed, no experiment uses it)"


def t_plotly():
    import plotly
    import plotly.graph_objects as go
    go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
    return f"plotly {plotly.__version__} at {where('plotly')} (installed, no experiment uses it)"


def t_ipywidgets():
    import ipywidgets as w
    s = w.FloatSlider(value=2.5, min=1, max=3.5)
    assert s.value == 2.5
    return f"ipywidgets {w.__version__} at {where('ipywidgets')}; FloatSlider"


def t_nbformat():
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("1+1"))
    nbformat.validate(nb)
    return f"nbformat {nbformat.__version__} at {where('nbformat')}; new notebook validates"


def t_nbclient():
    import nbclient
    from nbclient import NotebookClient  # noqa: F401
    import nbconvert
    import ipykernel
    return f"nbclient {nbclient.__version__}, nbconvert {nbconvert.__version__}, ipykernel {ipykernel.__version__} at {where('nbclient')}"


def t_manim():
    import manim
    from manim import Text, Axes, ValueTracker  # noqa: F401
    return f"Manim Community {manim.__version__} at {where('manim')}; Text/Axes/ValueTracker importable"


def t_femwell():
    import femwell
    from femwell.maxwell.waveguide import compute_modes  # noqa: F401
    from femwell.mesh import mesh_from_OrderedDict  # noqa: F401
    return f"femwell {version_of('femwell')} at {where('femwell')}; compute_modes, mesh_from_OrderedDict importable"


def t_skfem():
    import logging
    logging.getLogger("skfem").setLevel(logging.WARNING)
    import skfem
    from skfem import Basis, ElementTriP0, MeshTri
    m = MeshTri.init_symmetric().refined(2)
    b = Basis(m, ElementTriP0())
    assert b.zeros().shape[0] == m.t.shape[1]
    return f"scikit-fem {skfem.__version__} at {where('skfem')}; P0 basis on a refined MeshTri"


def t_gmsh():
    import gmsh
    gmsh.initialize()
    try:
        v = gmsh.option.getString("General.Version")
        gmsh.model.add("check")
        gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.mesh.generate(2)
        n = len(gmsh.model.mesh.getNodes()[0])
    finally:
        gmsh.finalize()
    return f"gmsh {v} at {where('gmsh')}; meshed a unit square ({n} nodes)"


def t_shapely():
    import shapely
    assert shapely.box(-0.25, -0.11, 0.25, 0.11).area == 0.5 * 0.22
    return f"shapely {shapely.__version__} at {where('shapely')}; box area of the 500x220 nm core"


def t_meshio():
    import meshio
    import numpy as np
    m = meshio.Mesh(np.array([[0, 0], [1, 0], [0, 1]], float), [("triangle", np.array([[0, 1, 2]]))])
    assert m.cells[0].data.shape == (1, 3)
    return f"meshio {meshio.__version__} at {where('meshio')}; one-triangle Mesh"


def t_tidy3d():
    import logging
    logging.getLogger("tidy3d").setLevel(logging.ERROR)
    import tidy3d as td
    td.config.logging.level = "ERROR"
    from tidy3d.plugins.mode import ModeSolver  # noqa: F401
    si = td.Medium(permittivity=3.5 ** 2)
    wg = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(td.inf, 0.5, 0.22)), medium=si)
    assert wg.medium.permittivity == 12.25
    return f"tidy3d {td.__version__} at {where('tidy3d')}; Medium/Box/Structure, ModeSolver importable"


def t_jax():
    import jax
    import jax.numpy as jnp
    v = float(jnp.abs(jnp.exp(-2j * jnp.pi * 2.5 * 39.6 / 1.31)))
    assert abs(v - 1) < 1e-5
    return f"jax {jax.__version__} at {where('jax')}; |exp(-j beta L)| = 1 on {jax.default_backend()}"


def t_sax():
    import sax
    import jax.numpy as jnp

    def coupler(coupling=0.107):
        k = coupling ** 0.5
        t = (1 - coupling) ** 0.5
        return sax.reciprocal({("in0", "out0"): t, ("in0", "out1"): -1j * k, ("in1", "out0"): -1j * k, ("in1", "out1"): t})

    def waveguide(wl=1.31, length=39.6, neff=2.5, ng=4.2, wl0=1.31, loss_db_cm=125.0):
        n = neff - (wl - wl0) * (ng - neff) / wl0
        amp = 10 ** (-loss_db_cm * length * 1e-4 / 20)
        return sax.reciprocal({("in0", "out0"): amp * jnp.exp(-2j * jnp.pi * n * length / wl)})

    ring, _ = sax.circuit(
        netlist={"instances": {"c": "coupler", "r": "waveguide"},
                 "connections": {"c,out1": "r,in0", "r,out0": "c,in1"},
                 "ports": {"in": "c,in0", "out": "c,out0"}},
        models={"coupler": coupler, "waveguide": waveguide})
    S = ring(wl=jnp.array([1.309, 1.310, 1.311]))
    T = float(jnp.abs(S["in", "out"][1]) ** 2)
    assert 0 <= T <= 1
    return f"sax {sax.__version__} at {where('sax')}; all-pass ring circuit evaluates (T(1310) = {T:.3f})"


def t_gdsfactory():
    import logging
    logging.disable(logging.WARNING)
    import gdsfactory as gf
    gf.gpdk.PDK.activate()
    c = gf.components.straight(length=10)
    assert len(c.ports) == 2
    return f"gdsfactory {gf.__version__} (kfactory {version_of('kfactory')}, klayout {version_of('klayout')}) at {where('gdsfactory')}; PDK activated, straight() has 2 ports"


def t_fdtd_pkg():
    import fdtd
    g = fdtd.Grid(shape=(10, 10, 1), grid_spacing=0.1e-6)
    assert g.shape[0] == 10
    return f"fdtd {version_of('fdtd')} at {where('fdtd')} (installed, no experiment uses it; 09 imports its own fdtd.py)"


def t_h5py():
    import h5py
    import numpy as np
    with h5py.File(io.BytesIO(), "w") as f:
        f["x"] = np.arange(3)
        assert f["x"][2] == 2
    return f"h5py {h5py.__version__} at {where('h5py')}; in-memory file (Meep can write .h5; no experiment reads one)"


def t_imageio():
    import imageio
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    return f"imageio {imageio.__version__}, imageio-ffmpeg {version_of('imageio-ffmpeg')} at {where('imageio')}; bundled ffmpeg {os.path.basename(exe)}"


def t_pillow():
    import PIL
    from PIL import Image
    im = Image.new("RGB", (4, 4), "white")
    assert im.size == (4, 4)
    return f"Pillow {PIL.__version__} at {where('PIL')}; Image.new"


def t_pandas():
    import pandas as pd
    assert pd.DataFrame({"a": [1, 2]})["a"].sum() == 3
    return f"pandas {pd.__version__} at {where('pandas')} (installed, no experiment uses it)"


def t_common():
    sys.path.insert(0, ROOT)
    from common import REF, use_style, SERIES, PALETTE  # noqa: F401
    from common.params import k0_per_um, freq_thz
    from common.units import db_per_cm_to_alpha_per_um
    assert abs(k0_per_um() - 4.796324661969149) < 1e-9
    assert abs(freq_thz() - 228.849) < 1e-2
    a = db_per_cm_to_alpha_per_um(REF.loss_db_cm_doped)
    return f"common/ imports: REF.lambda_nm = {REF.lambda_nm}, k0 = {k0_per_um():.4f} rad/um, alpha(125 dB/cm) = {a:.3e} /um"


CHECKS = [
    ("Python (.venv)", t_python),
    ("uv", t_uv),
    ("Homebrew", t_brew),
    ("micromamba", t_micromamba),
    ("Meep (.meep, subprocess)", t_meep),
    ("harminv (binary)", t_harminv_binary),
    ("ngspice", t_ngspice),
    ("ngspice batch run", t_ngspice_batch),
    ("ffmpeg", t_ffmpeg),
    ("ffprobe", t_ffprobe),
    ("jupyter + kernel", t_jupyter),
    ("manim CLI", t_manim_cli),
    ("LaTeX (absence)", t_no_latex),
    ("numpy", t_numpy),
    ("scipy", t_scipy),
    ("matplotlib", t_matplotlib),
    ("sympy", t_sympy),
    ("python-control", t_control),
    ("scikit-rf", t_skrf),
    ("plotly", t_plotly),
    ("ipywidgets", t_ipywidgets),
    ("nbformat", t_nbformat),
    ("nbclient/nbconvert/ipykernel", t_nbclient),
    ("manim (import)", t_manim),
    ("femwell", t_femwell),
    ("scikit-fem", t_skfem),
    ("gmsh", t_gmsh),
    ("shapely", t_shapely),
    ("meshio", t_meshio),
    ("tidy3d", t_tidy3d),
    ("jax", t_jax),
    ("sax", t_sax),
    ("gdsfactory", t_gdsfactory),
    ("fdtd (package)", t_fdtd_pkg),
    ("h5py", t_h5py),
    ("imageio", t_imageio),
    ("Pillow", t_pillow),
    ("pandas", t_pandas),
    ("common/ (repo)", t_common),
]


def main() -> int:
    print(f"photonics-simulations environment check\nrepo: {ROOT}\ninterpreter: {sys.executable}\n")
    t0 = time.time()
    for name, fn in CHECKS:
        check(name, fn)
    n_fail = sum(1 for _, ok, _, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail} OK, {n_fail} FAIL, {time.time() - t0:.1f} s total")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
