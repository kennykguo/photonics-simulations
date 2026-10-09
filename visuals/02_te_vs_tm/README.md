# TE versus TM in a silicon waveguide

The 500 nm x 220 nm silicon strip at 1310 nm, solved with a full-vector finite-element
mode solver (femwell), with every field component pulled out and plotted. Nothing about
the polarizations is drawn by hand: the field jumps at the faces, the longitudinal
components, and the effective indices are all outputs of the solve, and each is checked
against a closed-form result.

## Run it

Uses the repo-root `.venv` (`bash ~/photonics-simulations/visuals/setup.sh` if it does not exist yet).

```bash
cd ~/photonics-simulations/visuals/02_te_vs_tm && ../../.venv/bin/python run.py
```

About 8 minutes, most of it in the two geometry sweeps. Then:

```bash
open ~/photonics-simulations/visuals/02_te_vs_tm/out/te_vs_tm_poster.png
```

```bash
open ~/photonics-simulations/visuals/02_te_vs_tm/out/te_tm_travelling.mp4
```

## The one-paragraph answer

Quasi-TE has its electric field across the width, lying in the chip plane. At the two
long faces (top and bottom) that field is tangential, and tangential E is continuous
across a boundary, so it stays inside the silicon. Quasi-TM points across the thickness,
straight through those long faces. There the normal component must satisfy continuity
of D = n²E, so E jumps up by (n_Si/n_SiO2)² = 5.85 the moment it leaves the silicon.
The TM field is thrown into the silica above and below: lower effective index (2.16
versus 2.71), less power in the silicon (75 % versus 85 %), a longer tail, and in a
ring a different FSR and a different thermal shift. That is why the platform is built
for quasi-TE.

## What each file shows

| file | what to look for |
|---|---|
| `te_vs_tm_poster.png` | Everything on one page. Start here. |
| `1_cross_sections.png` | The two modes with the transverse E field as arrows. TE arrows lie flat across the width. TM arrows point up through the top face, and the dark bands just above and below the silicon are where the TM field is strongest: outside the core. |
| `2_six_components.png` | All six of Ex, Ey, Ez, Hx, Hy, Hz for both modes, with the share of the energy in each. Every one is non-zero, which is what "quasi" means. TE0 is 84.5 % Ex, but 15.4 % of its electric energy is in Ez, concentrated at the sidewalls. TM0 carries 42 % of its electric energy in Ez. |
| `3_boundary_jumps.png` | Line cuts through the centre in both directions for both modes, with D = n²E overlaid. Where the field is normal to a face it jumps by the measured 5.84 (theory 5.85) and D stays continuous. Where it is tangential, it does not jump at all (measured ratio 0.997). This is the single rule behind every difference between the modes. |
| `4_slab_and_capacitor.png` | The 1-D slab version, solved exactly: TE n_eff 2.988, TM 2.396, matching the notes. The TM tail decays in 109 nm against 80 nm for TE. Right: the series-capacitor picture, arrows to scale. |
| `5_mode_map.png` | n_eff of every guided mode against strip width and against thickness, coloured by TE fraction. TE and TM are identical at the two square cross-sections (220 x 220 and 500 x 500), and swap order when the strip is taller than it is wide: the field always prefers to lie along the long faces. |
| `6_capstone.png` | The numbers that matter for a ring: group index, FSR of the 39.6 µm ring, resonance shift per kelvin, and where each mode's field sits against the faces. |
| `te_tm_travelling.mp4` / `.gif` | The fields moving along the guide. Top: TE0 seen from above (Ex and Ez). Bottom: TM0 seen from the side (Ey and Ez). The arrows are drawn to true geometric scale, so you can see the field lines bending at the faces where the longitudinal component lives. |
| `results.json` | Every number above. |

## Key numbers

| | quasi-TE0 | quasi-TM0 |
|---|---|---|
| dominant field | Ex, in the chip plane | Ey, out of the chip |
| n_eff | 2.712 | 2.165 |
| n_g | 4.157 | 4.671 |
| power in the silicon | 85 % | 75 % |
| ∂n_eff/∂n_Si | 1.08 | 1.00 |
| FSR of the 39.6 µm ring | 10.43 nm | 9.28 nm |
| resonance shift | 63.2 pm/K | 52.2 pm/K |
| exact 1-D slab of the same thickness | 2.988 | 2.396 |

## Checks

- Normal-E jump at the sidewall (TE) and at the top face (TM), extrapolated to the face
  from both sides: 5.844 and 5.842, against (n_Si/n_SiO2)² = 5.854. Agreement 0.2 %.
- Tangential E across the other faces: ratio 0.997, i.e. continuous.
- The 1-D slab effective indices 2.988 (TE) and 2.396 (TM) match the values in the notes.
- The TE0 effective index 2.7120 matches the earlier femwell and Tidy3D results to the
  fourth decimal. TM0 is 2.165 here against 2.166 to 2.188 from the two solvers before;
  TM is the more mesh-sensitive mode because its field is concentrated on the large
  faces.
- Ez comes out 90° behind the transverse field and with the sign required by
  ∇·D = 0, which is a check on the phase convention of the solver output.

## Limits

- Uniform silica cladding on all sides. A real chip has buried oxide below and a
  different top cladding. With an asymmetric cladding, TE1 and TM0 no longer cross
  freely near 630 nm width: they mix into hybrid modes, which matters for tapers.
- Modes within 0.05 of the silica index are left out of the mode map, because their
  tails reach the edge of the 1 µm simulation box and their n_eff is not reliable.
- The 500 nm strip also guides a TE1 mode (n_eff about 1.8) at 1310 nm. It is in the
  mode map; the cross-section figures show only the two fundamental modes.
- The animation shows the solved mode profiles carried along z at their own n_eff.
  It is not a time-domain simulation.
