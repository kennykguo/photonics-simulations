"""Interactive 3-D (Plotly): drag to rotate, scroll to zoom, hover for values.

Writes out/explore_3d.html, a single self-contained file (works offline). The
four surfaces are the solved fields behind steps 1, 4, 6 and 9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cmt import H, W

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def core_lines(xc, color):
    x = [xc - W / 2, xc + W / 2, xc + W / 2, xc - W / 2, xc - W / 2]
    y = [-H / 2, -H / 2, H / 2, H / 2, -H / 2]
    return go.Scatter3d(x=x, y=y, z=[0] * 5, mode="lines", line=dict(color=color, width=6),
                        showlegend=False, hoverinfo="skip")


def main():
    D = dict(np.load(OUT / "cache" / "solved.npz"))
    xs, ys = D["xs"][::2], D["ys"][::2]
    X, Y = np.meshgrid(xs, ys)
    xa, xb = float(D["xa"]), float(D["xb"])
    inB = (np.abs(X - xb) <= W / 2) & (np.abs(Y) <= H / 2)
    uA, uB = D["uA"][::2, ::2], D["uB0"][::2, ::2]
    panels = [
        ("step 1: mode of guide A, u_A", uA / np.abs(uA).max(), "Blues"),
        ("step 4: left-over Δ_B·u_A (only inside B)", (uA * inB) / np.abs(uA * inB).max(), "Purples"),
        ("step 6: integrand u_B·Δ_B·u_A", (uA * uB * inB) / np.abs(uA * uB * inB).max(), "Oranges"),
        ("step 9: odd supermode e_- (E_x)", np.real(D["vM_Ex"][::2, ::2]) / np.abs(D["vM_Ex"]).max(), "RdBu_r"),
    ]
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "surface"}] * 2] * 2,
                        subplot_titles=[p[0] for p in panels], vertical_spacing=0.06, horizontal_spacing=0.02)
    for k, (title, F, cmap) in enumerate(panels):
        r, c = k // 2 + 1, k % 2 + 1
        fig.add_trace(go.Surface(x=X, y=Y, z=F, colorscale=cmap, showscale=False,
                                 cmid=0 if cmap == "RdBu_r" else None,
                                 hovertemplate="x %{x:.3f} µm<br>y %{y:.3f} µm<br>value %{z:.3f}<extra></extra>"),
                      row=r, col=c)
        fig.add_trace(core_lines(xa, "#1f5fa8"), row=r, col=c)
        fig.add_trace(core_lines(xb, "#c4501d"), row=r, col=c)
    scene = dict(xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title="field (norm.)",
                 aspectmode="manual", aspectratio=dict(x=2.4, y=1.0, z=0.7),
                 camera=dict(eye=dict(x=0.3, y=-2.0, z=1.1)))
    fig.update_layout(title="Coupled-mode theory: the solved fields in 3-D (drag to rotate)",
                      height=1100, width=1600, margin=dict(l=10, r=10, t=70, b=10),
                      scene=scene, scene2=scene, scene3=scene, scene4=scene)
    fig.write_html(OUT / "explore_3d.html", include_plotlyjs=True, full_html=True)
    print("  wrote explore_3d.html")


if __name__ == "__main__":
    main()
