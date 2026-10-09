# How two waveguides trade light: coupled-mode theory, one question per page

This follows the chain of causality in the notes, from one isolated guide to `dB/dz = −jκ_c A`
and the cos²/sin² power exchange, then checks it against a full 3-D Maxwell simulation.
Every picture is a solved field, never a sketch. The device is two 500 × 220 nm silicon
strips in silica, 150 nm apart, at 1310 nm, with light launched into guide A.

**The answer:** κ_c = 0.0240 rad/µm, so all the light is in guide B after
L_c = 65.6 µm (converged mode solver). A 3-D FDTD run with no coupled-mode theory in it
moves 99.8 % of the light into B, with L_c = 65.0 µm at 20 px/µm and 59.6 µm at 25 px/µm.
The FDTD length is not grid-converged; see Limits.

## Run it

Everything uses the repo-root environments (`bash ~/photonics-simulations/visuals/setup.sh`
creates `.venv` if it is missing; `env/setup.sh` at the repo root creates `.meep`).

```bash
cd ~/photonics-simulations/visuals/03_coupled_modes && ../../.venv/bin/python run.py
```

About 3 min the first time (8 min with the gap sweep), 3 min after that. To re-run the
3-D FDTD check (about 35 min at 20 px/µm and 80 min at 25 px/µm, single process):

```bash
cd ~/photonics-simulations/visuals/03_coupled_modes && ../../.meep/bin/python fdtd3d.py 20
```

## Look at it

```bash
open ~/photonics-simulations/visuals/03_coupled_modes/out/coupled_modes_walkthrough.pdf
```

```bash
open ~/photonics-simulations/visuals/03_coupled_modes/out/coupler_walk.mp4
```

```bash
open ~/photonics-simulations/visuals/03_coupled_modes/out/explore_3d.html
```

| file | what it is |
|---|---|
| `out/coupled_modes_walkthrough.pdf` | The 18 pages in order. Start here. |
| `out/pages/NN_stepSS_*.png` | The same pages as images. |
| `out/coupler_walk.mp4` / `.gif` | A walk along the coupler. Left: the cross-section at the current z. Right: the top view and the power in each guide, with a marker at the current z. |
| `out/explore_3d.html` | Four of the 3-D surfaces, rotatable in a browser (drag, scroll, hover for values). Works offline. |
| `out/renders/*.png` | The raw PyVista renders. |
| `out/results.json` | Every number on the pages. |

## The pages

Each page answers one question. The bar at the top shows where the page sits in the
chain. The numbered markers are explained in one line at the bottom, and the blue
"so →" line is the conclusion that leads to the next page. The fuller explanation of
each page is below.

**Steps 1 to 6 use the notes' scalar model.** It is solved exactly with finite
elements, so they show the derivation as written. **Step 7 compares it with full
Maxwell, and from then on the real (vector) numbers are used.**

1. **One guide** (page 1). The mode shape `u_A(x, y)` of guide A alone, drawn as a
   height above the cross-section. It comes from solving
   `∇²u + (k0²n² − β²)u = 0` with P2 finite elements (n_eff = 2.812 in the scalar
   model). The skirt outside the silicon is the evanescent tail. Along z only the phase
   `e^{−jβz}` changes.
2. **Add B** (page 2). The index map is a sum of masks:
   `n² = n_clad² + Δ_A + Δ_B`. Each Δ is 10.16 inside its core and 0 elsewhere. `u_A`
   was solved without `Δ_B`, so the equation changes only inside B.
3. **A's tail reaches B** (page 3). On a log scale, A's field visibly overlaps B's
   rectangle. Between the cores it falls by a factor e every 74 nm, and it is 4.4 % of
   its peak at B's near face. Only 0.023 % of A's |u|² lies inside B's silicon, but
   that is not zero.
4. **`u_A` is no longer exact** (pages 4 and 5).
   - Page 4: the three 3-D renders are `u_A × Δ_B = Δ_B·u_A`, the part of A's field
     sitting inside B's silicon.
   - Page 5 is a computation, not a drawing. `u_A` was plugged into the two-guide
     equation and whatever does not cancel was plotted. The leftover equals
     `k0²Δ_B u_A` to 1.00000 inside B, and is below 0.04 % of its peak everywhere
     else. This leftover is the notes' "source term".
5. **The field becomes a blend of two shapes** (pages 6 and 7).
   - Page 6: the exact field of the two-guide structure at five positions (grey) is
     matched by `|A|·u_A + |B|·u_B` (dashed). The shapes stay fixed; only the two
     weights change.
   - Page 7: the carrier repeats every 466 nm, while the weights change over 30 µm,
     65× more slowly. So `|A''|/|2βA'| = κ/2β = 0.0019`, and dropping A'' (the
     slowly varying envelope approximation) is justified.
6. **Project onto B** (pages 8 and 9).
   - Page 8: projection is a dot product. A's leftover is multiplied by B's own shape
     point by point; the volume under that surface is the overlap integral.
   - Page 9 is a finding the notes do not mention. A's leftover overlaps B's
     antisymmetric TE1 mode *more* than its TE0 mode (1.29×), because A's tail falls
     by 840× across B and is very lopsided. It overlaps TM0 hardly at all (6×10⁻⁵),
     because the field points the wrong way.
   - Overlap alone is therefore not enough. TE1 travels at n_eff = 1.81 instead of
     2.71, so its drive drifts out of step every 1.5 µm and cancels. At most 0.02 % of
     the power can build up in it (from κ²/(κ² + δ²)). Coupling needs both spatial
     overlap and phase matching.
7. **The number κ_c** (pages 10 and 11).
   - Inside its own scalar model, the notes' formula gives 0.05192 rad/µm. The exact
     answer, `(β₊ − β₋)/2` from solving both guides as one structure, is 0.05198
     (99.9 %). **The derivation is right.**
   - For the real strip (femwell, full Maxwell), the vector version of the formula
     gives 0.0236 against the exact 0.0240 (98.5 %). The scalar model, however, is
     **2.2× too strong**.
   - Page 11 shows why. Splitting the vector overlap by field component gives:
     - E_xE_x alone would give +0.063.
     - The longitudinal field E_z (17 % of |E|² in a silicon strip, the same
       component as in the TE/TM set) gives −0.039.
     - Inside B, A's field lines and B's own field lines tilt along z in opposite
       directions, so the E_zE_z term cancels most of the E_xE_x overlap.
   - The scalar model has no E_z, so it misses this cancellation.
8. **Power oscillates** (page 12).
   - The coupled equations, integrated numerically (lines), match the exact field
     (circles). The slow drift visible by 2L_c is the 1.5 % difference between the
     overlap κ and the exact κ.
   - Plotting A against jB gives a circle traversed at rate κ_c. This is the
     oscillator `A'' = −κ²A`, so `P_A = cos²κz` and `P_B = sin²κz`.
9. **Supermodes beat** (pages 13 and 14).
   - The true modes of the pair are e₊ (n = 2.7177) and e₋ (2.7077).
   - Light launched in A alone is 50.9 % e₊ and 49.1 % e₋.
   - The top view is the exact superposition. The two modes slip out of step, and
     half a slip moves the light to B after `L_c = λ/2(n₊ − n₋) = 65.6 µm`. This is
     the same κ as in step 7.
10. **3-D Maxwell check** (pages 15 and 16).
    - Meep steps Maxwell's equations in time on a 3-D grid with no coupled-mode theory
      inside. Guide B starts abruptly at z = 0, so the launch is exactly A = 1, B = 0.
    - Flux planes every 1 µm measure the power in each half of the cross-section. 99.8 %
      of the power reaches B, and P_A + P_B stays constant to 0.01 %.
    - The sin² fit gives L_c = 65.0 µm at 20 px/µm and 59.6 µm at 25 px/µm, against 65.6 µm
      from the converged mode solver. At 25 px/µm the FDTD tracks Meep's own mode solver on
      the same grid (dashed curve) to 95 %. The physics is reproduced; the exact length is
      still limited by the grid.
11. **Gap sensitivity** (pages 17 and 18).
    - κ_c against gap is a straight line on a log axis, with slope γ = 11.9 /µm. The
      overlap formula tracks the exact value at every gap from 100 to 400 nm.
    - The local decay rate of κ_c matches the decay rate of the *field* tail, not the
      power tail, which decays twice as fast. That is why κ ∝ e^{−γg} and not
      e^{−2γg}: the overlap integral contains u_A once.
    - A ±10 nm gap error changes κ_c by +13 % / −11 %. That turns a 50/50 splitter
      into 60/40 or 41/59, while a full-crossover coupler barely moves.

## Tools

| tool | what it does here |
|---|---|
| **scikit-fem** (P2 finite elements) | The notes' scalar wave equation, for each guide alone and for the pair. It is also used for the residual of page 5. |
| **femwell** (full-vector FEM, Nédélec + P2) | The real quasi-TE modes, the supermodes, the vector overlap and the TE1/TM0 projections. |
| **gmsh / shapely** | One shared mesh for A alone, B alone and the pair, so every overlap integral is taken on the same elements without interpolation. |
| **scipy** | Integrates the coupled-mode ODEs (`solve_ivp`), runs the eigen-solves and fits the FDTD sin² curve. |
| **Meep 1.34** (3-D FDTD, MIT) | The independent time-domain check. Its built-in MPB eigensolver gives the same-grid supermodes. |
| **PyVista / VTK** | The 3-D renders. White means no field, so only the field reads as ink, and labels are pinned to projected 3-D points. |
| **Plotly** | `explore_3d.html`, the rotatable 3-D surfaces. |
| **matplotlib + ffmpeg** (from imageio-ffmpeg) | The pages, the PDF and the video. |

## Checks

| check | result |
|---|---|
| Scalar model: overlap formula vs exact supermode splitting | 0.05192 vs 0.05198 rad/µm (99.9 %) |
| Full Maxwell: vector overlap formula vs exact supermode splitting | 0.0236 vs 0.0240 (98.5 %); 94 % to 99.8 % across 100 to 400 nm |
| Residual of u_A in the pair equation, inside B | equal to k0²Δ_B u_A to 1.00000; < 0.04 % of peak elsewhere |
| Isolated quasi-TE0 n_eff | 2.7120, the same as the TE/TM set and the earlier femwell and Tidy3D runs |
| Power conservation, exact supermode propagation | P_A + P_B constant to 1×10⁻⁵ |
| A rebuilt from e₊ and e₋ | 2.9 % residual (A is not exactly inside the span of the two supermodes, as expected) |
| 3-D FDTD, 20 / 25 px/µm | L_c 65.0 / 59.6 µm vs 65.6 µm converged; 99.8 % transfer; P_A + P_B constant to 0.01 % |
| 3-D FDTD vs Meep's mode solver on the same grid | κ 83 % (20 px/µm) → 95 % (25 px/µm) |

## Limits, stated plainly

- **FDTD resolution: not converged.** The 220 nm core is only 4.4 cells tall at 20 px/µm
  and 5.5 at 25 px/µm. Meep's mode solver on those grids gives n_eff = 2.58 and 2.64
  (converged: 2.71), and κ = 0.0290 and 0.0278 (converged: 0.0240). The FDTD's L_c went
  from 65.0 µm to 59.6 µm between the two grids, so the near-perfect match at 20 px/µm was
  partly luck: grid errors cancelling. The time-domain solve does confirm the physics:
  full transfer, the sin² shape and power conservation. It does not yet confirm the length
  to better than about 10 %. A converged value needs roughly 40 px/µm or more, a run of
  many hours on one core (MPI was slower than serial on this Mac). Any `fdtd3d.py <res>`
  output that is present is picked up by `run.py` automatically.
- **Cladding.** The silica is uniform on all sides. A real chip has buried oxide
  below and often a different top cladding.
- **Single wavelength.** Everything is at 1310 nm. κ_c grows with wavelength, because
  the tail reaches further.
- **Abrupt start.** Guide B starts abruptly in the FDTD, which is what makes the
  launch A = 1, B = 0. Real couplers bring B in with S-bends, which adds some
  coupling before and after the straight section.
- **Animation.** The walk along the coupler is the exact superposition of the two
  solved supermodes. For a straight section that is the exact steady-state field,
  but it is not a time-domain movie. Pages 15 and 16 are the time-domain solve.
