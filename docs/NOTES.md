# Kenny's notes: from Maxwell to the directional coupler

(These are the concepts every experiment in this repository is built to illustrate. Section numbers are referenced by the experiments' READMEs. Convention throughout: e^{jωt}; a +z wave carries e^{−jβz}.)

The central picture: a waveguide does not contain a little ray bouncing between walls. It supports particular electromagnetic field patterns, modes, that reproduce the same cross-sectional shape as they advance along the guide. The field oscillates in time, its phase advances along the guide, and its amplitude decays outside the core.

## 1. Notation and the physical geometry

Symmetric dielectric slab: core index n1 for −d < x < d, cladding n2 < n1 outside, uniform along y and z, propagation along z, x perpendicular to the boundaries. Symbols: E (V/m), H (A/m), D (C/m²), B (T), P (C/m²), ε0, μ0, ω = 2πf, λ0 vacuum wavelength, k0 = 2π/λ0 = ω/c, β longitudinal propagation constant (rad/m), F(x) transverse electric-field profile. Lowercase bold e(x,y) is the vector mode profile, not Euler's number; for the slab TE mode e(x,y) = ŷ F(x).

## 2. What a travelling wave means

E(z,t) = E0 cos(ωt − βz). Fixed z: sinusoidal oscillation in time. Fixed t: a spatial snapshot. Both varying: a sequence of snapshots shifting forward. Following a crest, ωt − βz = φ0, so v_p = dz/dt = ω/β (phase velocity). cos(ωt + βz) travels in −z. A guided mode keeps the same cross-sectional shape, multiplied by a common time- and z-dependent scalar.

## 3. Phasors: why time sometimes disappears

E(z,t) = Re{Ẽ(z) e^{jωt}} with Ẽ(z) = E0 e^{−jβz}. Writing E_y(x,z) = F(x)e^{−jβz} is the phasor; the physical field is Re{F(x) e^{−jβz} e^{jωt}} = F(x) cos(ωt − βz) for real F. Drop e^{jωt} only for single-frequency phasors; drop e^{−jβz} only when discussing the cross-sectional profile. Consistent pairs: e^{jωt}e^{−jβz} or e^{−jωt}e^{+jβz}. A pulse has many frequencies, so pulse analysis uses a frequency integral.

## 4. Wavevector, wavenumber, transverse and longitudinal

E(r,t) = E0 cos(ωt − k·r), k = k_x x̂ + k_y ŷ + k_z ẑ, direction normal to constant-phase surfaces, magnitude = phase change per metre. Uniform lossless material: |k| = n k0 = nω/c, λ = 2π/k = λ0/n. Longitudinal = along z (k_z = β); transverse = x-y. In a homogeneous core k_x² + β² = n1² k0². In the cladding k_x can become imaginary: exponential decay rather than a travelling direction.

## 5. Curl, divergence and the wave equation

Maxwell (SI): ∇·D = ρ_f, ∇·B = 0, ∇×E = −∂B/∂t, ∇×H = J_f + ∂D/∂t. Divergence = net outward flux per volume; curl = local circulation. Source-free uniform dielectric with D = εE, B = μH: curl of Faraday, Ampère, identity ∇×(∇×E) = ∇(∇·E) − ∇²E, and ∇·E = 0 give ∇²E − με ∂²E/∂t² = 0 (same for H). v = 1/√(με); nonmagnetic: v = c/√ε_r; define v = c/n so n² = ε_r.

## 6. Susceptibility and the complex refractive index

D = ε0E + P, P = ε0 χ̃_e E, so ε̃_r = 1 + χ̃_e and ñ² = 1 + χ̃_e. Passive material with e^{jωt}: ñ = n_r − jκ_ext, κ_ext > 0 (the notes' n'' is κ_ext). k̃ = ñ k0; e^{−jk̃z} = e^{−j n_r k0 z} e^{−κ_ext k0 z}: phase constant n_r k0, field-amplitude decay constant κ_ext k0. E(z,t) = E0 e^{−κ_ext k0 z} cos(ωt − n_r k0 z). Intensity I ∝ |E|², I(z) = I0 e^{−2κ_ext k0 z}, so α_abs = 2κ_ext k0 = 4πκ_ext/λ0 (1/m). The factor 2: amplitude is linear in E, power quadratic.

## 7. What "material response" means

Transfer-function language: input applied field, output polarization: χ̃_e(ω) = P̃(ω)/(ε0 Ẽ(ω)): how much polarization per field, its phase relative to the field, and the frequency dependence. Driven oscillator for one group of bound electrons: m_i ẍ_i + m_i γ_i ẋ_i + m_i ω_i² x_i = −q_i E(t). Phasors: m_i(−ω² + jγ_iω + ω_i²) x̃_i = −q_i Ẽ, so x̃_i = −(q_i/m_i) Ẽ/(ω_i² − ω² + jγ_iω). Dipole p̃_i = −q_i x̃_i = (q_i²/m_i) Ẽ/(…). With N_i oscillators per m³, P̃_i = N_i p̃_i, so χ̃_i(ω) = A_i/(ω_i² − ω² + jγ_iω) with A_i = N_i q_i²/(ε0 m_i) (units s⁻²). Multiple charge motions add: χ̃_e = Σ χ̃_i (contributions to the same polarization, not different waves).

## 8. Why a distant resonance still matters

For ω ≪ ω_i, χ̃_i ≈ A_i/ω_i² is not zero: the field still displaces the bound charge, quasi-statically, without the large amplitude and phase lag found near resonance (mass-spring below resonance). As ω approaches ω_i, ω_i² − ω² shrinks and the displacement per unit field becomes more frequency-sensitive; that gives a changing refractive index in a transparent region. A resonance affects the real index, dn/dλ0, and d²n/dλ0² far from resonance without strong absorption.

## 9. Deriving the Sellmeier form

Away from absorption (|ω_i² − ω²| ≫ γ_iω): χ_i ≈ A_i/(ω_i² − ω²). With ω = 2πc/λ0, ω_i = 2πc/λ_i: ω_i² − ω² = (2πc)²(λ0² − λ_i²)/(λ_i²λ0²), so χ_i(λ0) = [A_i λ_i²/(2πc)²] λ0²/(λ0² − λ_i²) = B_i λ0²/(λ0² − C_i), B_i = A_i λ_i²/(2πc)², C_i = λ_i². Hence n²(λ0) − 1 = Σ B_i λ0²/(λ0² − C_i): the Sellmeier equation. B_i dimensionless strengths, C_i wavelength squared, √C_i the fitted resonance wavelength; one term may represent a family of transitions. Coefficient tables in µm need λ0 in µm. Damping was discarded, so Sellmeier predicts the real index in transparent regions only.

## 10. Why ultraviolet and infrared resonances bend the curve differently

S_i(λ0) = B_i λ0²/(λ0² − λ_i²) contributes to n² − 1. UV resonance (λ0 ≫ λ_u): S_u ≈ B_u + B_u λ_u²/λ0², dS_u/dλ0 ≈ −2B_uλ_u²/λ0³ < 0, d²S_u/dλ0² ≈ 6B_uλ_u²/λ0⁴ > 0: decreasing, bending upward. IR resonance (λ0 ≪ λ_ir): S_ir ≈ −(B_ir/λ_ir²) λ0², dS_ir/dλ0 ≈ −2(B_ir/λ_ir²)λ0 < 0, d²S_ir/dλ0² ≈ −2B_ir/λ_ir² < 0: decreasing, bending downward. What cancels at the zero-dispersion wavelength is d²n/dλ0² ≈ 0, not dn/dλ0. Strictly n = √(1+S), dn/dλ0 = S'/(2n), d²n/dλ0² = S''/(2n) − (S')²/(4n³); the exact zero must be evaluated on the full n(λ0).

## 11. Why zero curvature means zero material dispersion

k(ω) = n(ω)ω/c; v_g = (dk/dω)⁻¹; dk/dω = (n + ω dn/dω)/c; n_g = n + ω dn/dω. With ω = 2πc/λ0, dλ0/dω = −λ0/ω, so n_g = n − λ0 dn/dλ0. Group delay per length τ_g/L = n_g/c. D_mat = d(τ_g/L)/dλ0 = (1/c) dn_g/dλ0 = −(λ0/c) d²n/dλ0² (SI s/m², usually ps/(nm·km)). Zero-dispersion: d²n/dλ0² ≈ 0. Example n = a + bλ0 gives dn/dλ0 = b but n_g = a: phase index varies, group index locally constant.

## 12. How a pulse is delayed and distorted

Analytic input ℰ_in(t) = ∫ A(Ω) e^{j(ω_c+Ω)t} dΩ = e^{jω_c t} × envelope. After length L each frequency gets e^{−jβ(ω_c+Ω)L}. Taylor: β(ω_c+Ω) ≈ β0 + β1Ω + ½β2Ω² with β1 = dβ/dω (s/m), β2 = d²β/dω² (s²/m). Output: e^{jω_c t − jβ0 L} ∫ A(Ω) e^{jΩ(t − β1L)} e^{−j(β2L/2)Ω²} dΩ. Constant term rotates the carrier phase; linear term is a pure delay t_g = β1 L = L/v_g; quadratic term gives frequency-dependent delay β2 L Ω (group-velocity dispersion) that distorts the pulse. β2 = 0 (and negligible higher orders over the bandwidth) means delay without reshaping.

## 13. What TE means

Transverse = perpendicular to z. TE: E_z = 0. Slab TE mode: E = ŷ F(x) e^{−jβz}; H has both H_x and H_z. A rectangular silicon waveguide supports a quasi-TE mode: mostly transverse electric field along one direction, with a small longitudinal component.

## 14. The waveguide problem: what is unknown?

Given n1, n2, 2d, ω, geometry; find F(x) and β. Complete TE field: E^real = Re{ŷ F(x) e^{jωt − jβz}}; for a lossless symmetric slab F real, E = ŷ F(x) cos(ωt − βz). The scale of F is arbitrary until the launched power is specified (linear homogeneous eigenproblem).

## 15. Why F(x)e^{−jβz} is allowed

Scalar TE wave equation ∂²Ẽ_y/∂x² + ∂²Ẽ_y/∂z² + n²(x)k0² Ẽ_y = 0; separation Ẽ_y = X(x)Z(z) gives X''/X + n²k0² = −Z''/Z = β² (constant, since one side depends on x only and the other on z only). Z'' + β²Z = 0 → Z = C+e^{−jβz} + C−e^{jβz}; forward mode Z = e^{−jβz}. Transverse equation: F'' + [n²(x)k0² − β²]F = 0. Deeper statement: translation invariance along z; exponentials are eigenfunctions of translation.

## 16. Solving the slab mode

Core: F'' + h²F = 0, h = √(n1²k0² − β²), F_core = A cos(hx) + B sin(hx) ("the field oscillates in the core" is a statement about shape in x at fixed t and z, not energy vanishing). Cladding: confined mode needs β > n2k0; γ = √(β² − n2²k0²) > 0; F'' − γ²F = 0; bounded solutions F_R = C_R e^{−γ(x−d)} for x > d and F_L = C_L e^{+γ(x+d)} for x < −d. Decay length δ = 1/γ: after one decay length the amplitude is e⁻¹ ≈ 0.368 of the boundary value, the intensity e⁻² ≈ 0.135.

## 17. Why people write k_x = jγ

In the cladding k_x² = n2²k0² − β² = −γ² so k_x = ±jγ. In e^{−jk_x x}: choosing k_x = −jγ on the right gives e^{−γx} (since −j(−j) = −1); choosing +jγ on the left gives e^{+γx}. Pick, on each side, the solution that stays finite far away. An imaginary transverse wavenumber means the transverse exponential changes from phase rotation to magnitude change.

## 18. Why the magnetic field appears in boundary matching

At a dielectric interface tangential E and tangential H are continuous (y and z are tangential to a plane of constant x). For E = ŷ F e^{−jβz}: ∇×E = x̂ jβF e^{−jβz} + ẑ F' e^{−jβz}; Faraday ∇×E = −jωμH gives H_x = −(β/ωμ) F e^{−jβz} and H_z = (j/ωμ) F' e^{−jβz}. Continuity of E_y: F_core = F_clad at the boundary. Continuity of H_z: (1/μ1)F'_core = (1/μ2)F'_clad; with μ1 ≈ μ2 ≈ μ0 this is F'_core = F'_clad. Derivative continuity comes from tangential H plus Faraday's law, not from an extra assumption.

## 19. Even and odd TE modes and the eigenvalue equations

Even: F_core = A cos(hx), F_R = C e^{−γ(x−d)}; continuity C = A cos(hd); derivatives −Ah sin(hd) = −γC; so h tan(hd) = γ. Odd: F_core = A sin(hx); C = A sin(hd); Ah cos(hd) = −γC; so −h cot(hd) = γ. h and γ both depend on β; only particular β satisfy the equation (eigenvalues); the corresponding F(x) are eigenfunctions. Several roots can exist: fundamental even mode with no transverse node, next odd with one node, and so on; a small guide may support only the fundamental.

## 20. What a mode specifically is

A self-consistent field pattern satisfying Maxwell's equations and all boundary conditions that preserves its transverse shape while propagating: Ẽ_m = e_m(x,y) e^{−jβ_m z}, H̃_m = h_m(x,y) e^{−jβ_m z}. "Modal" = associated with a mode (modal field, modal propagation constant, modal power, modal amplitude, multimode guide). An arbitrary launched field decomposes as Σ a_m e_m e^{−jβ_m z} + radiation fields + evanescent fields.

## 21. Standing across x, travelling along z

cos(hx) = ½(e^{jhx} + e^{−jhx}); the full phasor is ½A(e^{−j(βz − hx)} + e^{−j(βz + hx)}): two diagonal plane-wave components with transverse components ±h and the same longitudinal β. Their transverse power flows cancel; their longitudinal flows add. Fixed interference pattern across x, advancing along z. The physical field A cos(hx) cos(ωt − βz) scales and reverses sign as the travelling phase changes; its shape stays. A fundamental mode may hold less than half a cosine cycle in the core.

## 22. Effective index and longitudinal wavelength

n_eff = β/k0; β = n_eff 2π/λ0; bound slab mode n2 < n_eff < n1. Guided phase wavelength λ_g = 2π/β = λ0/n_eff; for λ0 = 1.310 µm and n_eff = 2.97, λ_g ≈ 0.441 µm. This is a phase wavelength: not the pulse length, not the decay distance, not the distance energy travels per modulation period. For a calculated mode n_eff = β/k0 is exact; the effective-index method (replacing a 2-D guide by slab problems) is approximate. n_eff alone gives phase propagation, not field distribution, group velocity, confinement, bend loss, or polarization. Group index n_g = c dβ/dω = n_eff + ω dn_eff/dω = n_eff − λ0 dn_eff/dλ0; v_g ≈ c/n_eff only if n_eff varies negligibly with frequency.

## 23. Interpreting the 220 nm slab description

2d = 220 nm, d = 110 nm. Fundamental even TE mode: F ~ cos(hx) inside, e^{−γ(|x|−d)} outside; nonzero at x = ±d (tangential E continuous; dielectric boundaries do not force E_y = 0, perfect conductors do). Decay length 80 nm means γ⁻¹ ≈ 80 nm. Whole profile carries the phase e^{−jβz}, repeating every λ_g. A real rectangular SOI guide needs a full-vector numerical mode solution.

## 24. Total internal reflection and the evanescent field

Snell n1 sin θ_i = n2 sin θ_t; critical sin θ_c = n2/n1; beyond it no propagating transmitted wave, but not zero field: tangential E and H must stay continuous and the tangential wavevector is shared (k_z = β). In the low-index medium k_x² + β² = n2²k0² with β > n2k0 gives k_x = ±jγ: E_ev ∝ e^{−γx} e^{jωt − jβz}, decaying away from the interface but advancing in phase along it. A waveguide is the two-interface version. The ray picture is useful; the mode picture is fundamental and valid for subwavelength cores.

## 25. The Poynting vector and the evanescent-field statement

S(t) = E × H (W/m²); time-averaged ⟨S⟩ = ½ Re{Ẽ × H̃*}. Evanescent TE field in the cladding x > 0: Ẽ_y = E_a e^{−γx} e^{−jβz}, H̃_x = −(β/ωμ)Ẽ_y, H̃_z = −(jγ/ωμ)Ẽ_y. ⟨S_x⟩ = ½Re{Ẽ_y H̃_z*} = ½Re{(jγ/ωμ)|Ẽ_y|²} = 0 (purely imaginary); ⟨S_z⟩ = −½Re{Ẽ_y H̃_x*} = (β/2ωμ)|Ẽ_y|² > 0. So at one lossless interface the evanescent field carries no time-averaged power outward normal to the interface but can carry power parallel to it. Stored energy density ⟨u⟩ = ¼(ε|Ẽ|² + μ|H̃|²) decays with x; the normal Poynting component is reactive. Evanescent confinement (transverse decay, lossless ε, energy stays electromagnetic) versus material absorption (decay along propagation, ε'' > 0, energy becomes heat): with ε̃ = ε' − jε'', p_abs = ½ωε''|Ẽ|² > 0 in a passive absorber; an evanescent field in a lossless cladding has no such term.

## 26. Ports and modal power

A port is a chosen cross-sectional surface together with a chosen set of modes describing fields crossing it. Expand Ẽ = Σ s_m e_m, H̃ = Σ s_m h_m (incoming/outgoing separately when needed); s_m is a complex modal amplitude. Modal power P_m = ½ Re ∫_A (e_m × h_m*)·ẑ dA (W). With modes normalized to 1 W, P_m = |s_m|² (s_m in √W). Scattering matrices act on amplitudes, s_out = S s_in, not on powers: |s1 + s2|² = |s1|² + |s2|² + 2Re{s1 s2*}; the cross term is interference.

## 27. How an evanescent directional coupler works

Two identical guides close enough that their tails overlap; the second guide intercepts the near field (frustrated total internal reflection) and converts it to propagating modal power. With a(z), b(z) the modal amplitudes and κ_c the coupling coefficient (1/m), after removing the common phase: da/dz = −jκ_c b, db/dz = −jκ_c a. Then a'' + κ_c² a = 0; with a(0) = a0, b(0) = 0: a = a0 cos(κ_c z), b = −j a0 sin(κ_c z); P1 = |a0|² cos², P2 = |a0|² sin², P1 + P2 = |a0|²: power oscillates between guides. Supermode picture: even (same sign) and odd (opposite sign) supermodes with β+ and β−; launching guide 1 excites both equally; their beating moves the field between guides; κ_c = (β+ − β−)/2. Coupler S-matrix for length L_c: t = cos(κ_c L_c), K = sin(κ_c L_c), [s3; s4] = [[t, −jK], [−jK, t]] [s1; s2], |t|² + |K|² = 1; 50/50: |t| = |K| = 1/√2; the −j is a 90° phase shift and makes S unitary (S†S = I). Evanescent coupling alone is broadband; narrow frequency selection needs frequency-dependent phase accumulation and repeated interference (microring physics).

## 28. Round-trip retention factor a (definition only)

P(L) = P(0) e^{−α_abs L}; |A(L)| = |A(0)| e^{−α_abs L/2}. For a closed path of length L_rt: a = e^{−α_abs L_rt/2}; A_after = a e^{−jβL_rt} A_before (a = field retention per trip, the exponential the phase). Retained power fraction a² = e^{−α_abs L_rt}. Some authors define a as a power factor; in the usual microring field equations a multiplies field amplitude.

## The complete mental model

Time oscillation e^{jωt}; longitudinal propagation e^{−jβz}; transverse mode shape F(x); core confinement (sinusoidal in core, exponential in cladding); a mode = an allowed (F(x), β) pair; evanescent decay e^{−γx} confines without dissipating; absorptive decay e^{−αz/2} transfers energy to the material; modal power = longitudinal Poynting flux integrated over the cross-section; a port = a cross-section with modal amplitudes; a directional coupler = two guides whose overlapping tails make even/odd supermodes whose beating transfers power.
