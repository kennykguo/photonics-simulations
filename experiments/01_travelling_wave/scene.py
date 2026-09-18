"""Manim scenes for experiment 01 (rendered by run.py with `manim -qm`).

No LaTeX on this machine: all labels are Text() with Unicode.
Units on screen: z in nm, t in fs. Numbers come from common.REF via twave.py.

Scene 1  TravellingWaveScene  (~14 s)
    left  : snapshot E(z) at the current time, one crest tracked by a dot; the crest
            moves at v_p = w/b  (NOTES 2)
    right : the phasor  E~(z0) e^{jwt}  at the probe point z0, rotating counter-clockwise;
            its real-axis projection is the physical field at z0  (NOTES 3)
    bottom: the time trace E(z0, t) being drawn, period T = 4.37 fs
Scene 2  SignConventionScene  (~9 s)
    cos(wt - bz) and cos(wt + bz) side by side with tracked crests: one moves toward +z,
    the other toward -z  (NOTES 2, 3)
"""
import sys
import pathlib
import numpy as np
from manim import (
    Scene, Axes, Text, Dot, Arrow, DoubleArrow, Circle, DashedLine, Line, VGroup,
    ValueTracker, always_redraw, UP, DOWN, LEFT, RIGHT, linear, config,
)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from twave import wave_numbers, E_real  # noqa: E402
from common import PALETTE, SERIES  # noqa: E402  (twave.py put the repo root on sys.path)

W = wave_numbers()                      # 1310 nm, n_eff 2.5
OMEGA = W["omega_rad_s"]                # rad/s
BETA = W["beta_rad_m"]                  # rad/m
T_FS = W["T_s"] * 1e15                  # 4.37 fs
LAM_G_NM = W["lambda_med_m"] * 1e9      # 524 nm
VP = W["v_p_m_s"]                       # m/s

BLUE, ORANGE, AQUA, VIOLET, RED = SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[4]
INK, MUTED = PALETTE["ink"], PALETTE["muted"]

config.background_color = PALETTE["surface"]
# The z window is exactly three guided wavelengths so that a crest leaving the right
# edge re-enters at the left edge ON a crest (the wrap below is modulo Z_MAX_NM).
Z_MAX_NM = 3 * LAM_G_NM                 # 1572 nm


def field_nm_fs(z_nm, t_fs, sign=-1):
    return E_real(z_nm * 1e-9, t_fs * 1e-15, OMEGA, BETA, sign)


class TravellingWaveScene(Scene):
    def construct(self):
        t = ValueTracker(0.0)   # time in fs
        z0_nm = 250.0           # probe point for the phasor and the time trace

        title = Text("E(z,t) = E₀ cos(ωt − βz)      λ₀ = 1310 nm, n_eff = 2.5", font_size=28,
                     color=INK).to_edge(UP, buff=0.25)
        sub = Text(f"λ_g = λ₀/n_eff = {LAM_G_NM:.0f} nm     T = {T_FS:.2f} fs     "
                   f"v_p = ω/β = c/n_eff = {VP/1e8:.3f}×10⁸ m/s",
                   font_size=20, color=PALETTE["ink2"]).next_to(title, DOWN, buff=0.12)

        # ------------------------------------------------ left: snapshot in z
        ax = Axes(x_range=[0, Z_MAX_NM, 500], y_range=[-1.3, 1.3, 1], x_length=7.2, y_length=2.6,
                  axis_config={"color": MUTED, "include_numbers": True, "font_size": 18, "label_constructor": Text,
                               "decimal_number_config": {"num_decimal_places": 0, "color": MUTED}},
                  tips=False)
        ax.to_edge(LEFT, buff=1.0).shift(UP * 0.05)   # room under the axes for the λ_g label
        xl = Text("z (nm)", font_size=18, color=MUTED).next_to(ax.x_axis, RIGHT, buff=0.1)
        yl = Text("E / E₀", font_size=18, color=MUTED).next_to(ax.y_axis, LEFT, buff=0.05)
        wave = always_redraw(lambda: ax.plot(lambda z: field_nm_fs(z, t.get_value()),
                                             x_range=[0, Z_MAX_NM, 5], color=BLUE))

        # one crest: at t=0 the crests sit at z = m*lambda_g; follow the one at m=1
        def crest_z():
            # crest m=1 sits at z = lambda_g at t = 0 and moves at v_p; fs*m/s -> nm is 1e-6
            z = LAM_G_NM + VP * t.get_value() * 1e-6
            return z % Z_MAX_NM
        crest = always_redraw(lambda: Dot(ax.c2p(crest_z(), 1.0), color=ORANGE, radius=0.09))
        crest_lbl = Text("crest: ωt − βz = const → z = (ω/β) t", font_size=18,
                         color=ORANGE).next_to(ax, UP, buff=0.05).align_to(ax, RIGHT)
        probe = always_redraw(lambda: Dot(ax.c2p(z0_nm, field_nm_fs(z0_nm, t.get_value())),
                                          color=VIOLET, radius=0.08))
        probe_line = DashedLine(ax.c2p(z0_nm, -1.3), ax.c2p(z0_nm, 1.3), color=VIOLET,
                                stroke_width=1.5)
        probe_lbl = Text(f"z₀ = {z0_nm:.0f} nm", font_size=16, color=VIOLET).next_to(
            ax.c2p(z0_nm, 1.3), UP, buff=0.05)
        # crest-to-crest distance: the bracket is anchored to the tracked crest and moves
        # with it, so at every instant its two ends sit on neighbouring crests (the
        # earlier static bracket only spanned crest to crest at t = m*T).
        def bracket_ends():
            zc = crest_z()
            # crest_z is in [0, Z_MAX_NM); pick the neighbour that stays inside the window
            if zc + LAM_G_NM <= Z_MAX_NM:
                return zc, zc + LAM_G_NM
            return zc - LAM_G_NM, zc
        lam_arrow = always_redraw(lambda: DoubleArrow(
            ax.c2p(bracket_ends()[0], -1.15), ax.c2p(bracket_ends()[1], -1.15), buff=0,
            color=AQUA, stroke_width=2.5, tip_length=0.15))
        # label under the bracket (below the troughs), never on the curve
        lam_lbl = always_redraw(lambda: Text(
            f"λ_g = {LAM_G_NM:.0f} nm, crest to crest", font_size=14, color=AQUA,
        ).next_to(lam_arrow, DOWN, buff=0.03))

        # ------------------------------------------------ right: phasor at z0
        ph_center = np.array([4.3, 0.55, 0])
        R = 1.15
        circ = Circle(radius=R, color=MUTED, stroke_width=1.5).move_to(ph_center)
        re_axis = Line(ph_center + LEFT * (R + 0.3), ph_center + RIGHT * (R + 0.3), color=MUTED,
                       stroke_width=1.2)
        im_axis = Line(ph_center + DOWN * (R + 0.3), ph_center + UP * (R + 0.3), color=MUTED,
                       stroke_width=1.2)
        re_lbl = Text("Re", font_size=16, color=MUTED).next_to(re_axis, RIGHT, buff=0.05)
        im_lbl = Text("Im", font_size=16, color=MUTED).next_to(im_axis, UP, buff=0.05)

        def phase():
            return OMEGA * t.get_value() * 1e-15 - BETA * z0_nm * 1e-9

        def tip():
            return ph_center + R * np.array([np.cos(phase()), np.sin(phase()), 0])
        arrow = always_redraw(lambda: Arrow(ph_center, tip(), buff=0, color=VIOLET,
                                            stroke_width=4, max_tip_length_to_length_ratio=0.15))
        proj = always_redraw(lambda: DashedLine(tip(), np.array([tip()[0], ph_center[1], 0]),
                                                color=VIOLET, stroke_width=1.5))
        proj_dot = always_redraw(lambda: Dot(np.array([tip()[0], ph_center[1], 0]),
                                             color=VIOLET, radius=0.07))
        ph_title = Text("phasor  Ẽ(z₀)·e^(jωt) = E₀ e^(−jβz₀) e^(jωt)", font_size=18,
                        color=INK).next_to(circ, UP, buff=0.45)
        ph_note = Text("rotates counter-clockwise at ω;\nRe{·} = E(z₀,t)", font_size=16,
                       color=PALETTE["ink2"], line_spacing=0.8).next_to(circ, DOWN, buff=0.35)

        # ------------------------------------------------ bottom: time trace at z0
        T_END = 3 * T_FS
        ax_t = Axes(x_range=[0, T_END, T_FS], y_range=[-1.3, 1.3, 1], x_length=7.2, y_length=1.7,
                    axis_config={"color": MUTED, "include_numbers": False}, tips=False)
        ax_t.to_edge(LEFT, buff=1.0).to_edge(DOWN, buff=0.4)
        ticks = VGroup(*[Text(f"{k*T_FS:.2f}", font_size=14, color=MUTED).next_to(
            ax_t.c2p(k * T_FS, -1.3), DOWN, buff=0.08) for k in range(4)])
        tl = Text("t (fs)", font_size=16, color=MUTED).next_to(ax_t.x_axis, RIGHT, buff=0.1)
        tt = Text(f"E(z₀, t): fixed z → oscillation in time, period T = {T_FS:.2f} fs",
                  font_size=16, color=VIOLET).next_to(ax_t, UP, buff=0.02)

        def trace():
            tv = max(t.get_value(), 1e-3)
            return ax_t.plot(lambda tt_: field_nm_fs(z0_nm, tt_), x_range=[0, tv, 0.02],
                             color=VIOLET)
        trace_m = always_redraw(trace)
        clock = always_redraw(lambda: Text(f"t = {t.get_value():5.2f} fs", font_size=22,
                                           color=INK).to_corner(DOWN + RIGHT, buff=0.35))

        self.add(title, sub, ax, xl, yl, wave, probe_line, probe_lbl, probe, crest, crest_lbl,
                 lam_arrow, lam_lbl, circ, re_axis, im_axis, re_lbl, im_lbl, arrow, proj, proj_dot, ph_title, ph_note,
                 ax_t, ticks, tl, tt, trace_m, clock)
        self.wait(0.5)
        # run for three periods; the crest travels 3 lambda_g = 1572 nm = one full window
        self.play(t.animate.set_value(T_END), run_time=12.0, rate_func=linear)
        self.wait(0.8)


class SignConventionScene(Scene):
    def construct(self):
        t = ValueTracker(0.0)
        title = Text("Sign convention: with e^{jωt}, the sign of βz sets the direction",
                     font_size=26, color=INK).to_edge(UP, buff=0.3)

        def panel(sign, color, label, y_shift):
            ax = Axes(x_range=[0, Z_MAX_NM, 500], y_range=[-1.3, 1.3, 1], x_length=9.0,
                      y_length=1.9,
                      axis_config={"color": MUTED, "include_numbers": True, "font_size": 16, "label_constructor": Text,
                                   "decimal_number_config": {"num_decimal_places": 0, "color": MUTED}},
                      tips=False).shift(UP * y_shift)
            wave = always_redraw(lambda: ax.plot(lambda z: field_nm_fs(z, t.get_value(), sign),
                                                 x_range=[0, Z_MAX_NM, 5], color=color))

            def crest_z():
                # crest of cos(wt + sign*bz): sign*b z = -wt + 2 pi m
                z = 2 * LAM_G_NM - sign * VP * t.get_value() * 1e-6   # fs*m/s -> nm: 1e-15*1e9
                return z % Z_MAX_NM                                   # window = 3 lambda_g
            dot = always_redraw(lambda: Dot(ax.c2p(crest_z(), 1.0), color=ORANGE, radius=0.09))
            lbl = Text(label, font_size=20, color=color).next_to(ax, UP, buff=0.02).align_to(ax, LEFT)
            zl = Text("z (nm)", font_size=16, color=MUTED).next_to(ax.x_axis, RIGHT, buff=0.1)
            return VGroup(ax, wave, dot, lbl, zl)

        # double space after "+z" / "−z": Manim Text() kerning otherwise swallows the single space ("+zat")
        p1 = panel(-1, BLUE, f"cos(ωt − βz): crest moves toward +z  at v_p = +{VP/1e8:.3f}×10⁸ m/s", 1.05)
        p2 = panel(+1, RED, f"cos(ωt + βz): crest moves toward −z  at −{VP/1e8:.3f}×10⁸ m/s", -1.75)
        note = Text("Consistent pairs: e^(jωt) e^(−jβz) (forward)  or  e^(−jωt) e^(+jβz) (same wave, other convention)",
                    font_size=18, color=PALETTE["ink2"]).to_edge(DOWN, buff=0.25)
        clock = always_redraw(lambda: Text(f"t = {t.get_value():5.2f} fs", font_size=20,
                                           color=INK).to_corner(UP + RIGHT, buff=0.3).shift(DOWN * 0.6))
        self.add(title, p1, p2, note, clock)
        self.wait(0.4)
        self.play(t.animate.set_value(2 * T_FS), run_time=8.0, rate_func=linear)
        self.wait(0.6)
