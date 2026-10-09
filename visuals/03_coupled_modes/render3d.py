"""3-D renders with PyVista (VTK). Every render is a solved field, drawn as a
height surface over the waveguide cross-section or as slices through the coupler.

Each function writes a PNG and returns the pixel position of named 3-D anchor
points, so the matplotlib page that embeds the render can put labels exactly on
the features they describe.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
from matplotlib import colormaps

pv.OFF_SCREEN = True
WIN = (1500, 1000)
A_COL, B_COL = "#1f5fa8", "#c4501d"

from matplotlib.colors import LinearSegmentedColormap

# Light at zero, so "no field" reads as white paper and the field reads as ink.
CMAP_A = LinearSegmentedColormap.from_list("ink_blue", ["#f7f9fc", "#b9d3ec", "#3d7cc0", "#0b2e5c"])
CMAP_B = LinearSegmentedColormap.from_list("ink_orange", ["#fcf8f5", "#f0c2a6", "#d0632b", "#6b2208"])
CMAP_P = LinearSegmentedColormap.from_list("ink_purple", ["#faf8fc", "#d4c4e8", "#7b52b3", "#341a5c"])


def _project(pl, pts):
    """World (x, y, z) -> image pixel (col, row from top)."""
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    out = {}
    h = pl.window_size[1]
    for name, p in pts.items():
        coord.SetValue(*p)
        x, y = coord.GetComputedDoubleDisplayValue(pl.renderer)
        out[name] = (x, h - y)
    return out


def _core_box(pl, xc, w, h, zlen=None, z0=0.0, color="#888888", opacity=0.25, zscale=1.0, flat=False):
    """A silicon core: a translucent prism (flat=True: a thin slab under a height plot)."""
    if flat:
        box = pv.Box(bounds=(xc - w / 2, xc + w / 2, -h / 2, h / 2, -0.004, 0.0))
    else:
        box = pv.Box(bounds=(xc - w / 2, xc + w / 2, -h / 2, h / 2, z0, z0 + zlen))
    pl.add_mesh(box, color=color, opacity=opacity, show_edges=False, smooth_shading=False)
    pl.add_mesh(box.extract_feature_edges(), color="#333333", line_width=2)


def height_surface(path, X, Y, F, cores, *, zscale=0.45, cmap="RdBu_r", clim=None,
                   anchors=None, camera=None, mask=None, floor=False, colors=None, fmax=None,
                   core_style="box"):
    """F(x, y) drawn as a surface z = zscale * F / max|F| above the cross-section.

    cores: list of (x_centre, w, h, edge_colour). mask: optional bool array; where
    False the surface is drawn transparent-grey (used to show 'only this part
    survives')."""
    pl = pv.Plotter(off_screen=True, window_size=WIN)
    pl.set_background("white")
    fmax = fmax or np.nanmax(np.abs(F)) or 1.0
    Z = zscale * F / fmax
    grid = pv.StructuredGrid(X, Y, Z)
    grid["f"] = (F / fmax).T.ravel() if False else (F / fmax).ravel(order="F")
    lim = clim if clim is not None else (-1, 1) if cmap in ("RdBu_r", "PuOr_r") else (0, 1)
    pl.add_mesh(grid, scalars="f", cmap=cmap, clim=lim, smooth_shading=True,
                show_scalar_bar=False, specular=0.25, ambient=0.25)
    # a coarse wireframe on the surface itself: the depth cue
    st = max(1, int(round(0.05 / (X[0, 1] - X[0, 0]))))
    wire = pv.StructuredGrid(X[::st, ::st], Y[::st, ::st], Z[::st, ::st] + 1e-4)
    pl.add_mesh(wire.extract_all_edges(), color="#5a6470", opacity=0.22, line_width=1)
    if mask is not None:
        ghost = pv.StructuredGrid(X, Y, Z + 0.0)
        ghost["m"] = (~mask).astype(float).ravel(order="F")
        pl.add_mesh(ghost.threshold(0.5, scalars="m"), color="#bbbbbb", opacity=0.18)
    # contour lines on the surface, for depth cues
    try:
        pl.add_mesh(grid.contour(isosurfaces=np.linspace(lim[0], lim[1], 11)[1:-1], scalars="f"),
                    color="#222222", line_width=1, opacity=0.35)
    except Exception:
        pass
    if floor:
        x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
        fl = pv.Plane(center=((x0 + x1) / 2, (y0 + y1) / 2, -0.004), direction=(0, 0, 1),
                      i_size=x1 - x0, j_size=y1 - y0)
        pl.add_mesh(fl, color="#ffffff", opacity=0.0)
        for gx in np.arange(np.ceil(x0 * 4) / 4, x1 + 1e-9, 0.25):
            pl.add_mesh(pv.Line((gx, y0, -0.003), (gx, y1, -0.003)), color="#c9d0d8", line_width=1)
        for gy in np.arange(np.ceil(y0 * 4) / 4, y1 + 1e-9, 0.25):
            pl.add_mesh(pv.Line((x0, gy, -0.003), (x1, gy, -0.003)), color="#c9d0d8", line_width=1)
    for xc, w, h, col in cores:
        if core_style == "outline":   # flat outline: never hides a negative lobe
            pts = np.array([[xc - w / 2, -h / 2, 0], [xc + w / 2, -h / 2, 0], [xc + w / 2, h / 2, 0],
                            [xc - w / 2, h / 2, 0], [xc - w / 2, -h / 2, 0]], float)
            pl.add_mesh(pv.lines_from_points(pts), color=col, line_width=5)
            continue
        b = pv.Box(bounds=(xc - w / 2, xc + w / 2, -h / 2, h / 2, -0.003, 0.035))
        pl.add_mesh(b, color=col, opacity=0.45)
        pl.add_mesh(b.extract_feature_edges(), color=col, line_width=3)
    cam = camera or dict(position=(0.25, -2.75, 1.55), focal_point=(0.0, 0.05, 0.10), viewup=(0, 0, 1))
    pl.camera.position, pl.camera.focal_point, pl.camera.up = cam["position"], cam["focal_point"], cam["viewup"]
    pl.camera.view_angle = 24
    pl.enable_anti_aliasing("ssaa")
    pl.render()
    px = _project(pl, anchors or {})
    pl.screenshot(path, transparent_background=False)
    pl.close()
    return px


def coupler(path, xs, ys, field_at, zs_slices, z_len, *, zscale_len, cores, top_I=None,
            top_z=None, anchors=None, camera=None):
    """The coupler in 3-D, z (propagation) compressed by zscale_len.

    field_at(z) -> |E|^2 on the (y, x) grid xs, ys. The mid-height plane shows
    top_I (|E|^2 at y = 0 along z, rows = x) as a colour map; vertical slices
    show the full cross-section at zs_slices."""
    pl = pv.Plotter(off_screen=True, window_size=(1700, 1000))
    pl.set_background("white")
    cm = colormaps["inferno"]
    vmax = max(field_at(z).max() for z in zs_slices)
    # mid-height plane: x across, z along (drawn as the 3rd axis, compressed)
    if top_I is not None:
        Xp, Zp = np.meshgrid(xs, top_z * zscale_len)
        plane = pv.StructuredGrid(Xp, np.zeros_like(Xp), Zp)
        plane["I"] = (top_I.T / top_I.max()).ravel(order="F")
        pl.add_mesh(plane, scalars="I", cmap="inferno", clim=(0, 1), show_scalar_bar=False,
                    opacity="linear", ambient=0.6)
    for z in zs_slices:
        I = field_at(z) / vmax
        Xs, Ys = np.meshgrid(xs, ys)
        sl = pv.StructuredGrid(Xs, Ys, np.full_like(Xs, z * zscale_len))
        sl["I"] = I.ravel(order="F")
        # dim = transparent, bright = opaque: each slice shows only where the light is
        pl.add_mesh(sl, scalars="I", cmap="inferno_r" if False else "inferno", clim=(0, 1),
                    show_scalar_bar=False, ambient=0.8, opacity=[0.0, 0.35, 0.8, 1.0, 1.0])
        x0, x1, y0, y1, zz = xs.min(), xs.max(), ys.min(), ys.max(), z * zscale_len
        pts = np.array([[x0, y0, zz], [x1, y0, zz], [x1, y1, zz], [x0, y1, zz], [x0, y0, zz]], float)
        pl.add_mesh(pv.lines_from_points(pts), color="#9aa3ad", line_width=1.5)
    for xc, w, h, col in cores:
        b = pv.Box(bounds=(xc - w / 2, xc + w / 2, -h / 2, h / 2, 0, z_len * zscale_len))
        pl.add_mesh(b, color=col, opacity=0.06)
        pl.add_mesh(b.extract_feature_edges(), color=col, line_width=2.5)
    cam = camera or dict(position=(-6.5, 3.2, -1.2), focal_point=(0, 0, z_len * zscale_len / 2),
                         viewup=(0, 1, 0))
    pl.camera.position, pl.camera.focal_point, pl.camera.up = cam["position"], cam["focal_point"], cam["viewup"]
    pl.camera.view_angle = 30
    pl.enable_anti_aliasing("ssaa")
    pl.render()
    px = _project(pl, anchors or {})
    pl.screenshot(path, transparent_background=False)
    pl.close()
    return px
