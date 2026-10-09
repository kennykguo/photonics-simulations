# Seeing n_eff and n_g in a waveguide

A silicon waveguide carries one mode, and that one mode has two different speeds.
This folder computes both for the 500 nm x 220 nm silicon strip at 1310 nm and then
shows you the difference, including an animation where you can literally watch the
wave crests overtake the pulse they belong to.

| | meaning | this waveguide |
|---|---|---|
| **n_eff = beta / k0** | the **phase** index. Wave crests move at c/n_eff. | **2.7120**, so crests move at 110.5 um/ps |
| **n_g = n_eff - lambda dn_eff/dlambda** | the **group** index. The envelope, the energy and the information move at c/n_g. | **4.1568**, so the pulse moves at 72.1 um/ps |

The crests run **1.53 times faster than the pulse**. Nothing is typed in: the material
indices come from Sellmeier fits, the mode comes from a full-vector finite-element
solve, and n_g comes from differentiating that solve with respect to wavelength.

## Set up (fresh machine)

```bash
cd ~/photonics-simulations && bash visuals/setup.sh   # once
```

That makes a `.venv` with numpy, scipy, matplotlib, femwell (the mode solver) and a
bundled ffmpeg. It takes about two minutes and installs nothing globally.

## Run it

```bash
cd ~/photonics-simulations/visuals/01_neff_and_ng && ../../.venv/bin/python run.py
```

About 90 seconds. It prints the headline numbers and writes everything to `out/`.

## See the results

```bash
open ~/photonics-simulations/visuals/01_neff_and_ng/out
```

Or open them one at a time:

```bash
open ~/photonics-simulations/visuals/01_neff_and_ng/out/phase_vs_group.mp4
open ~/photonics-simulations/visuals/01_neff_and_ng/out/2_phase_vs_group.png
```

The `.gif` is the same animation if you want something that plays in any viewer or
pastes into a document.

## What each file shows

**`phase_vs_group.mp4`** (8 s, and `phase_vs_group.gif`) is the one to watch first.
A pulse travels down the guide. The orange dot rides the envelope peak at c/n_g. The
blue dot is glued to one particular wave crest and moves at c/n_eff. The crest starts
1 um *behind* the peak and ends 2.84 um *ahead* of it: over 100 fs it gains 3.84 um,
which is 8 guided wavelengths. Crests are born at the back of the pulse and die at the
front. That gap is the whole difference between the two indices.

**`2_phase_vs_group.png`** is the same thing frozen at three times, plus a
distance-versus-time plot where the two speeds are just two different slopes.

**`1_mode_and_indices.png`** is the mode itself. Panel (a) is the real 2-D mode from
the FEM solver, 99.3 % TE. Panel (b) is the exact 1-D slab solution for the same
220 nm thickness, which you can solve by hand: it gives 2.988 because it has no side
walls, and the real strip comes out lower at 2.712 because it is also squeezed
sideways. Panel (c) is the readout, including the number the capstone cares about:
the mode's grip on the silicon, dn_eff/dn_Si = 1.08, which turns 1 K of heating into
a 63 pm resonance shift.

**`3_where_ng_comes_from.png`** is the graphical meaning of the formula. Plot
n_eff against wavelength, draw the tangent at 1310 nm, extend it back to
lambda = 0, and the intercept *is* n_g = 4.157. The right panel shows both indices
across the O band; the gap between them is exactly the -lambda dn_eff/dlambda term,
which is 1.44 here.

**`4_width_sweep.png`** shows that both indices belong to the *shape*, not just the
material. Widen the guide from 300 to 750 nm and n_eff climbs toward the infinitely
wide slab value of 2.988 while n_g falls. They move in opposite directions, which is
the clearest proof that they are not two names for the same thing.

**`5_ring_capstone.png`** is why this matters for the Lightmatter work. In a 39.6 um
ring, n_eff decides **where** a resonance sits (m = n_eff L / lambda = 81.98, so order
82 puts one at 1309.7 nm) and n_g decides **how far apart** they are
(FSR = lambda^2/(n_g L) = 10.43 nm). Use n_eff for the spacing by mistake and you get
15.98 nm, 53 % too wide, and your channel plan is wrong.

**`results.json`** has every number the figures quote, including both sweeps.

## Things to try

Edit the constants at the top of `run.py` and re-run.

1. `HEIGHT = 0.34` (a thicker silicon layer). The mode is better confined, n_eff rises
   toward the bulk silicon index, and the gap to n_g narrows.
2. `WIDTH = 0.32`. Close to the single-mode cutoff: n_eff drops toward the cladding
   index, the mode spills into the silica, and n_g climbs toward 4.6 (it is 4.63 at 300 nm)
   because the mode redistributes much faster with wavelength.
3. `LAM = 1.55` (the C band). Run it and compare: the same guide is less confining at
   a longer wavelength.
4. In the animation block, set `Z_CREST0 = 4.0` to tag a crest ahead of the peak
   instead of behind it, and watch it run out of the front of the pulse.
5. Set `n_core` and `n_clad` by hand in `indices.strip_modes` (in the top folder) to a low-contrast pair
   like 1.50 and 1.45 (a glass fibre rather than a silicon wire). The two indices
   nearly coincide, which is why the phase/group distinction rarely bites in fibre and
   always bites in silicon.

## What is doing the work

- **femwell** (with scikit-fem, gmsh, shapely) solves the full-vector Maxwell
  eigenproblem on a triangular mesh. It is the source of n_eff and of the mode picture.
  A solve takes about 0.3 s after the first one.
- **scipy** bisects the slab's transcendental eigenvalue equation for the exact 1-D
  check, and **numpy** does the Sellmeier fits, the derivatives and the fields.
- **matplotlib** draws everything and, with the ffmpeg binary that ships inside
  **imageio-ffmpeg**, encodes the animation. No Homebrew ffmpeg needed.

## Checks and honest limits

- n_eff = 2.7120 and n_g = 4.1568 reproduce the values from the earlier
  [photonics-simulations](https://github.com/kennykguo/photonics-simulations) work,
  where femwell and Tidy3D agreed with each other to 0.06 %.
- The mode's sensitivity dn_eff/dn_Si = 1.0774 was verified two ways, by femwell's
  confinement functional and by re-solving with the silicon index nudged by +/- 0.01.
  They agree to 0.00 %.
- That sensitivity is 1.08, not the 0.85 power-confinement factor, which is why this
  gives 63 pm/K where the textbook's rule of thumb gives 50 pm/K. The difference is
  real and it matters for the capstone's loop gain.
- n_g is a central difference over +/- 20 nm. Tightening it to +/- 10 nm moves n_g by
  less than 0.001.
- The cladding is uniform silica here. A real chip has a buried oxide below and a
  different top cladding, which shifts n_eff by a few times 0.01.
- The animation is a 1-D scalar cartoon of the field along z. It uses the real
  n_eff and n_g but does not re-solve Maxwell in time; experiment 10 of
  [photonics-simulations](https://github.com/kennykguo/photonics-simulations) does
  that with Meep FDTD if you want the honest version.
