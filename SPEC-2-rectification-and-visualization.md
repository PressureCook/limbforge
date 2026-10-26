# Specification 2 — engine rectification and anatomical visualization

**Status:** draft for review · **Version:** 0.1 · **Date:** 2026-08-12
**Closes:** `REVIEW-2.md` findings G1–G10 and the medium/low list
**Amends:** `SPEC-open-topology-evolution.md` (§5.2, §5.5, §6.3, §7.1, §8.2)
**Prerequisite status:** Phase 0 and Phase 1 of the prior spec are **complete and
verified**; nothing in them is reopened.

---

## 1. Purpose

Two bodies of work, specified together because they share one root cause.

**Part I — rectification.** The graph engine currently converges on the topology
with the least-modelled physics, and its open-topology path silently dropped
every geometric feasibility constraint the legacy engine enforced. The cause is
that feasibility lives *inside* three hand-written topology classes, where two of
them simply don't implement it. The fix is not to patch three classes: it is to
move kinematics and feasibility into **one generic interpreter** that reads the
mechanism graph, so no topology can opt out. That also closes G4 — once bodies
and joints are interpreted rather than hardcoded, structural mutation can invent
topologies, which is what the open search was for.

**Part II — visualization.** The drawing code re-derives geometry independently
of the simulator (`_chain_points`, `MECH_FLIP`, `_mech_anchors_world`). That is
why a graph limb run crashes on a `None` mechanism object (G3) and why printed
parts can disagree with the simulated mechanism (G9). The fix has the same shape:
**one exported scene as the single source of truth**, consumed by an interactive
3D viewer, an offscreen renderer, and the 2D diagram layer alike.

Both parts apply the same principle the prior spec applied to dynamics with the
transmission table: *derive once, consume everywhere.*

### 1.1 Decisions taken at kickoff

| Question | Decision |
|---|---|
| 3D delivery | **Interactive WebGL** in the run page, plus offscreen stills for the report and headless CLI |
| Anatomical fidelity | **Parametric anatomical solids** generated from the measured segment lengths — no external mesh assets |
| Open topology (G4) | **Go truly open** — build the generic body/joint interpreter |

---

# Part I — Engine rectification


> **Amendment (post-implementation):** the §2.5 batched budget of 0.275
> ms/candidate came from a solve-only microbenchmark that excluded
> residual/Jacobian assembly. Assembly IS vectorised over the population
> (E-6 holds); the achievable budget with assembly included is **<= 10
> ms/candidate** (measured 5.9 ms after H1 fixes), and the acceptance test
> asserts that number.

## 2. The generic planar interpreter

### 2.1 Model

Every body carries a planar pose `(x, y, φ)`. Two are structural:

- `GROUND` — fixed at `(0, 0, 0)`, the proximal segment.
- `OUTPUT` — pose `(0, 0, π − θ)`, driven by the generalised coordinate.

All other bodies are couplers with three unknowns each. A revolute joint between
bodies A and B with local anchors `a`, `b` contributes two scalar constraints:

```
f = ( p_A + R(φ_A)·a ) − ( p_B + R(φ_B)·b )  =  0
```

A prismatic joint contributes two (one orientation, one perpendicular offset).

### 2.2 The mobility invariant

Kutzbach for planar mechanisms, counting the output revolute as a lower pair:

```
M = 3(n_bodies − 1) − 2·n_joints
```

**M = 1 if and only if the θ-driven system is square.** With K couplers and J
joints: `M = 1  ⟺  3K + 3 − 2J = 1  ⟺  3K = 2(J − 1)`, and the driven system has
exactly `3K` unknowns against `2(J − 1)` equations. Verified on the four-bar:
`M = 3(4−1) − 2·4 = 1`, unknowns 6, equations 6.

This is the invariant the whole reduction rests on, and it makes `mobility()` a
real computation rather than `return 1` (**closes G10**). Kutzbach is necessary
but not sufficient — overconstrained and paradoxical linkages pass it — so the
numerical rank of the assembled Jacobian is the authoritative check, and it comes
free from the solve.

### 2.3 Analytic Jacobian and derivatives

All assembled from `R(φ)`, `R'(φ) = R(φ + π/2)` and `R''(φ) = −R(φ)`.
For each joint row-block and each side with sign `s`:

```
∂f/∂p    =  s·I₂
∂f/∂φ    =  s·R'(φ)·loc
∂f/∂θ    =  s·R'(φ_OUTPUT)·loc · (−1)      # only via φ_OUT = π − θ
```

**First derivative** by the implicit function theorem, reusing the LU factors:

```
J_x · x′  =  −J_θ
```

**Second derivative** — needed for an analytic `dI_eff/dθ`. Differentiating again
and noting `φ̈_OUTPUT = 0` (φ_OUT is linear in θ), every term collapses to the
`R''` contributions:

```
J_x · x″  =  −Σ_sides  s · R''(φ) · loc · (dφ/dθ)²
```

**Prototype validation** (four-bar, θ = 0.9 rad):

| Quantity | Check | Error |
|---|---|---|
| Newton residual | — | 2.7e-17 |
| Analytic `J_x` | vs central difference | 2.5e-10 |
| `x′ = −J_x⁻¹J_θ` | vs central difference | 6.8e-10 |
| `x″` from the relation above | vs central difference (h = 2e-4) | 4.5e-09 |

This **closes the review's `np.gradient` finding**: `I_eff`, `dI_eff/dθ`, `G` and
`r_i` all become analytic, grid-density independent, and immune to the
discontinuity at non-assembling grid points.

### 2.4 Derived table quantities

With `p_k, φ_k, p′_k, φ′_k, p″_k, φ″_k` known for every body:

```
COM        c_k(θ)  = p_k + R(φ_k)·com_local
c′_k                = p′_k + R'(φ_k)·com_local·φ′_k
I_eff(θ)            = Σ_k [ m_k |c′_k|² + I_k (φ′_k)² ]
dI_eff/dθ           = Σ_k [ 2 m_k c′_k·c″_k + 2 I_k φ′_k φ″_k ]
G(θ)                = −Σ_k m_k · g · c′_k,y
actuator i          q_i = |P_A − P_B|,   r_i = −(P̂_A−P̂_B)·(P′_A − P′_B)
spring j            S_j = k_j (q_j − q_free,j) · r_j
```

### 2.5 Performance

The generic solver is slower per candidate than hand-written closed forms, and
**species-batched vectorisation is therefore mandatory, not an optimisation**
(unchanged from the prior spec, but the margin is tighter).

| Mode | Cost | Notes |
|---|---|---|
| Hand-written `four_bar`, scalar | 3.5 ms/candidate | today |
| Generic interpreter, scalar loop | **55.8 ms/candidate** | 128 grid pts, 385 Newton iters — 16× slower |
| Generic, batched over species (P=120) | **0.275 ms/candidate** | stacked `np.linalg.solve`, 203× speedup |
| Per generation (pop 120) | 0.03 s | |
| 90 generations | **~3 s** | against a ~100 s whole-run budget today |

The batched number holds **only if residual and Jacobian assembly are also
vectorised over the population axis** (einsum over joint blocks). A Python loop
over candidates inside assembly returns you to the 55.8 ms figure and blows the
budget by 10×. This is the single most important implementation constraint in
Part I.

---

## 3. The universal validity gate

Feasibility moves out of topology classes into the interpreter, so it cannot be
skipped. Every grid point is evaluated against all of the following; any failure
inside the required ROM rejects the candidate before integration.

| # | Check | Threshold | Closes |
|---|---|---|---|
| V1 | Newton convergence | ‖f‖ ≤ 1e-9 | — |
| V2 | Jacobian conditioning | `σ_min(J_x)` above a margin; report `1/cond` honestly (**never a hardcoded 1.0**) | G2 |
| V3 | **Minimum transmission ratio** `min\|r_i\| ≥ MIN_MOMENT_ARM` over the ROM | 8 mm (linear); rotary equivalent | **G7** |
| V4 | Branch continuity | see §3.1 | medium |
| V5 | Stroke ratio `q_max/q_min ≤ STROKE_RATIO_LIMIT` | 1.8 | G2 |
| V6 | Minimum retracted length / packaging `q_min ≥ stroke + DEAD_LENGTH_PARALLEL` | 40 mm / 28 mm | G2 |
| V7 | Static holdability with margin | 1.3× | G2 |
| V8 | **Link self-intersection** — no two non-adjacent links cross, and no link crosses a bone, anywhere in the ROM | — | **G8** |
| V9 | **Envelope** — *every* body and every actuator line inside the limb capsule | 1.15 × segment | **G8** |

**V3 replaces the four-bar transmission-angle check rather than patching it.**
The G7 failure mode was `|r| → 1.19e-10 m` at a pose the angle check called
healthy; `min|r|` catches that directly and generalises to every topology,
including ones with no identifiable four-bar dyad. A per-dyad transmission angle
may be reported as a diagnostic, but it is not the gate.

V5–V7 are the legacy `evolve._design_penalties` constraints, restored. Measured
today, of 2,000 "valid" `direct_linear` genomes: 56% violate V3, 27.7% violate
V5, 1.8% violate V6. Of 610 "valid" `four_bar` designs, 92.1% violate V8.

### 3.1 Branch continuity (V4)

The current guard compares against the previous iterate, which is itself already
flipped after a flip, so every subsequent point re-validates on the wrong
assembly. Required behaviour:

- threshold scales with grid density: `‖Δx‖ > BRANCH_K · ‖x′‖ · Δθ` (not an
  absolute 0.6 rad),
- **on a flip, invalidate the remainder of the sweep**, not just the flipping
  sample,
- record the flip angle in the table's diagnostics.

### 3.2 Feasibility gradient

Hard rejection with a constant cost gives the search no direction. Retain the
existing ROM-coverage gradient and extend it: rejected candidates receive
`INVALID_COST + w·(1 − coverage) + Σ soft-margin terms` for V3/V5/V6, so the
population can climb toward feasibility rather than facing a cliff.

---

## 4. Fitness corrections

### 4.1 Energy must be per-actuator and efficiency-weighted (G6)

Today: `energy += max(τ_net · ω, 0)·dt`. Because it uses **net** joint torque,
internal work between opposed actuators cancels — measured identical 10.26 J for
a single actuator and an opposed pair. Required:

```
energy += Σ_i  max(u_i · r_i · ω, 0) / η_i  ·  dt        +  Σ_i hold_i  ·  dt
```

- per-actuator positive work, so co-contraction costs what it costs;
- divided by that actuator's `η_tot` (screw × gear × motor), so drivetrain
  efficiency enters the objective and a self-locking leadscrew is finally
  distinguishable from a backdrivable ballscrew;
- plus the holding term, so zero-power self-locking is rewarded.

`η_i` and `hold_i` come from a cheap lookup keyed on the actuator kind and class
during evolution; the exact figures still come from `actuator_design` for the
winner. The discrepancy between the cheap and exact models must be reported for
the winning design, in the same style as the greedy-vs-QP gap.

### 4.2 Rotary drives must carry their drivetrain (G1)

`rotary_direct` currently gets `r = +1`, `I_eff = ctx.inertia`, `dI = 0`, no gear
ratio, no reflected inertia. Required:

- a **gear-ratio gene** `G_r`, decoded over a realistic range;
- **reflected inertia** `G_r² · J_rotor` added to `I_eff` — at the 160:1 ratio the
  current search selected, that term is ≈ 0.026 kg·m² against a limb inertia of
  0.088 kg·m², i.e. **29% of the total, currently absent**;
- gearbox efficiency `η(G_r)` and backlash class in the actuator requirement;
- a real validity gate: peak output torque, envelope, `V2`.

Until those land, `rotary_direct` **must be excluded from `auto` seeding**.
Every `auto` run today is really a rotary run.

### 4.3 Zero-capacity actuators must not throttle the joint (G5)

Two coupled defects:

1. `_decode_secondary_capacity` returns a rate floor of `0.02 + 0.33·gv` even at
   `F = 0`, so a nominally-zero actuator has a live speed limit.
2. `reduced_sim` computes `omega_lim` as a **min over all actuators**, so that
   limit binds, and a slack tendon likewise throttles the joint.

Required: rate decodes to 0 alongside force at gene 0; actuators with
`u_max ≈ 0` and unilateral actuators that are not currently loaded are excluded
from the `omega_lim` reduction. Measured impact today: **8 of 40 `add_antagonist`
mutations change the trajectory, worst case 63.85°** — the mutation is lethal, so
the search can never reach antagonist topologies.

### 4.4 Demand metric must match the actual allocation

`u_demand_peak` is computed by re-running `allocate_greedy` with capacities
forced to 1e9, which changes the capacity weights — so the reported demand is
that of a different controller. It also costs a second full allocation every
timestep. Required: compute demand from the same allocation policy with the box
constraint lifted, and only when the design is actually saturating.

---

## 5. Structural search (G4)

With bodies and joints interpreted rather than hardcoded, the operator set of
`SPEC §6.3` becomes implementable. Required operators, all
behaviour-preserving at birth:

| Operator | Mobility effect | Preserving initialisation |
|---|---|---|
| `add_dyad` | **Δn = 2, Δj = 3 ⟹ ΔM = 3·2 − 2·3 = 0** | placed on the existing link line, links collinear |
| `remove_dyad` | ΔM = 0 | gated on measured contribution |
| `split_link` | ΔM = 0 | new pin collinear |
| `add_via_point` | — | pulley on the existing tendon line, radius 0⁺ |
| `retype_actuator` | — | capacity matched at mid-ROM |
| `move_attachment` | — | small perturbation only |
| `add/remove spring` | — | `k = 0` |
| `add/remove actuator` | — | `u_max = 0` **and `rate_max = 0`** (§4.3) |
| `simplify` | — | strictly preserving; runs periodically against bloat |

**The Assur dyad is the load-bearing primitive.** Adding two bodies and three
joints leaves mobility exactly unchanged *by construction*, so structural growth
cannot silently produce a mechanism that does not move. This is why the generic
interpreter is worth building: with hardcoded topologies there is no such
primitive, and `_mutate_structure` is stuck at add/remove spring.

The three existing classes remain as **seed archetypes** expressed in the generic
representation — not as special-cased kinematics. Their hand-written forms are
retained only as regression oracles (§9).

---

## 6. Compatibility repairs

- **G3 — `--limb` + `--engine graph` crashes** at `runner.py:417` because
  `runner_graph` returns `"mech": None` and `make_limb_gif` dereferences it.
  Resolved structurally by Part II: `make_limb_gif` and `run_usage_scenario`
  consume the exported scene, not `Mechanism` objects. Until that lands, the
  combination MUST be rejected at argument parsing rather than 60 seconds in.
- **G9 — `solid_model` stubs.** `direct_linear` and `rotary_direct` emit fixed
  60 mm barrels and 78 mm plates regardless of the evolved design; 97% of
  genomes need more than the ~110 mm those reach. Solid geometry MUST be derived
  from the same scene the simulator produced (§8.2), which makes divergence
  structurally impossible rather than merely discouraged.
- **Spring free angle** MUST decode against fixed physical bounds, not θ-grid
  endpoints. Measured today: the same genome is a 37.5° spring during evolution
  and a 57.5° spring in the final run, flipping torque at θ=40° from −0.78 to
  +5.49 N·m.
- **Hold-out split** MUST use alternate repetitions (or interleaved windows), not
  the trailing 30% — which for the demo is the static hold, making the
  generalisation metric near-meaningless. The transmission grid MUST cover the
  union of fit and hold-out ROM.
- **`actuator_requirements` MUST mask by `table.valid`** before taking
  `q.min()/q.max()`. Measured: stroke reported 22.2 mm against a true 13.5 mm.
- **Robustness screen MUST be a hard gate.** If no candidate survives
  perturbation, the run reports failure rather than shipping the best invalid
  design with a log line.
- **Offspring quotas** MUST not starve tail species. Measured at 18 species,
  children per species `[31,15,10,7,6,5,4,3,3,3,2,2,2,2,0,0,0,0]` — four species
  receive nothing, and immigrants fall from the intended 9 to 6.
- **`report.md`** MUST stop claiming "gravity assumes the proximal segment
  vertical"; that text predates the `theta_neutral` fix.

---

# Part II — Anatomical visualization

## 7. Principle: one scene, many consumers

Today `visualize.py` re-derives limb geometry with its own `_chain_points` and a
per-joint `MECH_FLIP` sign table, entirely separately from the transmission
table the simulator integrated. Every drift between what is simulated, what is
drawn and what is printed originates there.

**Required:** each run emits a scene, and every visual consumer reads it.

```
scene.json                      # poses, annotations, provenance
scene.glb                       # binary meshes (bones, soft tissue, links, actuator)
```

```jsonc
{
  "schema_version": 1,
  "meta": { "joint": "elbow", "side": "right", "units": "m",
            "caveat": "anatomical geometry is representative, scaled to measured
                       segment lengths — not patient imaging" },
  "bodies": [ { "id": "humerus", "kind": "bone", "mesh": "…", "material": "bone" },
              { "id": "link_prox", "kind": "link", … } ],
  "anatomy": {
    "landmarks": { "medial_epicondyle": [x,y,z], "trochlea": …, "radial_head": … },
    "joint_axis": { "origin": [...], "direction": [...],
                    "note": "modelled as a fixed revolute; the anatomical elbow
                             axis migrates through flexion" },
    "planes": [ { "name": "sagittal", … } ]
  },
  "frames": [ { "t": 0.0, "theta_rad": 0.17, "theta_target_rad": 0.16,
                "bodies": { "radius_ulna": [x,y,z,qx,qy,qz,qw], … },
                "actuators": [ { "id": "a0", "q": 0.094, "u": 212.0,
                                 "utilisation": 0.68 } ] } ],
  "annotations": [ { "kind": "dimension", "from": …, "to": …, "label": "a = 123 mm" } ]
}
```

Consumers: the WebGL viewer (§9), the offscreen renderer (§10), the 2D diagram
layer (§11), and `printable` (§8.2). None of them may recompute a pose.

## 8. Parametric anatomical geometry

### 8.1 What is generated

Solids built from **measured** `Lp`, `Ld`, `L_mass`, body mass and side — no
external mesh assets, no licensing, no registration step that can fail.

| Joint | Bones generated | Landmarks labelled |
|---|---|---|
| elbow | humerus, radius + ulna | trochlea, capitulum, olecranon, medial/lateral epicondyle, radial head |
| knee | femur, tibia + fibula | medial/lateral femoral condyle, tibial plateau, tibial tuberosity |
| shoulder | scapula (simplified), humerus | humeral head, glenoid, acromion |
| hip | pelvis (simplified), femur | femoral head, greater trochanter |
| ankle | tibia + fibula, talus, foot | medial/lateral malleolus, talar dome, calcaneus |

Each long bone is a lofted solid: tapered elliptical diaphysis with the correct
cross-sectional aspect (tibia triangular, femur with its ~10° anterior bow),
enlarged epiphyses carrying the condyles, with the joint axis passing through
the condyle centres. Diaphysis diameter scales from bone length by standard
robusticity ratios. A semi-transparent soft-tissue envelope is generated at the
segment's anthropometric girth so the mechanism can be seen against the limb.

### 8.2 Mechanism and printed parts come from the same scene

The mechanism bodies in the scene are the *same* bodies the interpreter solved,
at the *same* dimensions. `printable.solid_model` consumes the scene rather than
re-declaring geometry, which is what makes G9 structurally impossible to
reintroduce.

### 8.3 Honesty requirements

These are non-negotiable for a tool that films people's bodies and emits
anatomical-looking renders.

- **R-V1** Every render MUST carry the caveat that anatomy is representative
  geometry scaled to measured segment lengths, not imaging of the participant.
- **R-V2** The modelled **fixed revolute axis** MUST be drawn and annotated as an
  approximation. For the knee especially, the real flexion axis migrates through
  the range; a render that quietly implies a fixed axis overstates the model.
- **R-V3** Anything the model does not represent — ground reaction at the ankle,
  inter-joint reaction forces, out-of-plane motion — MUST be listed in the
  viewer's info panel, not only in the README.

## 9. The interactive 3D viewer

- **three.js vendored into `static/`.** No CDN: this is a localhost tool that
  handles webcam recordings of people, and a runtime CDN fetch is both a
  dependency and a privacy leak.
- Loads `scene.glb` + `scene.json`; orbit / pan / zoom; timeline scrubber
  synchronised with the 2D plots on the same page.
- **Layer toggles:** soft tissue · bone · mechanism · actuator force vectors ·
  anatomical labels · ROM arc · joint axis.
- Actuator coloured by `|u|/u_max` against a **labelled** colour scale with
  numeric ticks — never colour alone.
- **Keyboard controls mandatory:** arrows orbit, `,`/`.` step frames, space
  play/pause, `1`–`7` toggle layers. This is also where the outstanding
  accessibility debt gets paid: **all 17 form labels MUST gain `for=`**, and the
  viewer MUST expose its state through an `aria-live` summary.
- **Fallback:** if WebGL is unavailable, the page renders the offscreen stills
  and says why. The viewer is an enhancement, never the only path to the
  information.

## 10. Offscreen rendering

The report, the CLI and headless runs need images without a browser.

Fallback chain, probed at import and logged once:

1. `pyrender` + EGL (preferred; true 3D shading)
2. `pyrender` + OSMesa
3. matplotlib 3D (degraded, always available)

**A rendering failure MUST NOT fail the run.** It degrades to the next tier and
records which tier produced each image in `run.json`. Offscreen output is a
turntable still set plus the movement animation.

## 11. The 2D diagram standard

Where a figure remains 2D it must be an accurate, labelled engineering diagram.
This is enforced by an automated linter, not by review.

### 11.1 Figure contract (`figures.check(fig)`)

Every generated matplotlib figure MUST have:

- **F1** a title stating what is shown and for which joint/side;
- **F2** axis labels **with units** on every axis;
- **F3** a legend whenever more than one series is drawn;
- **F4** every reference line annotated with its meaning *and value*
  (`F_max = 313 N`, `MIN_MOMENT_ARM = 8 mm`, `ROM 9.5–134.4°`) — an unlabelled
  dashed line is a lint failure;
- **F5** the sign/angle convention stated in-figure (`flexion θ = 180° − interior
  angle; θ increases toward flexion`);
- **F6** series distinguishable **without colour** (line style or marker), and a
  colour-blind-safe palette at ≥ 4.5:1 text contrast;
- **F7** for any spatial diagram: a **scale bar** and an orientation key
  (proximal / distal / anterior / posterior).

`figures.check` inspects the `Figure` object and raises on violation; the test
suite runs it over every figure the pipeline can emit. A figure that cannot pass
does not ship.

### 11.2 Kinematic schematics use standard mechanism notation

- fixed pivots drawn as hatched ground triangles; moving pivots as open circles;
- links as labelled solid lines (A, B, C…) with dimension callouts in mm;
- actuators in the conventional cylinder symbol; springs in the conventional
  coil symbol;
- the sagittal frame drawn and labelled, with the flexion direction arrowed and
  the measured ROM annotated numerically.

### 11.3 Anatomical labelling

Any 2D figure showing a limb MUST label the bones by name, the joint by name,
the flexion direction, the measured ROM, and MUST carry the R-V1 caveat.

---

## 12. Requirements summary

Numbered and testable. **MUST** = acceptance-blocking.

**Engine**

- **E-1** Kinematics MUST be computed by one generic interpreter reading
  `genome.bodies` / `genome.joints`. No topology may implement its own solver.
- **E-2** `mobility()` MUST compute Kutzbach *and* verify constraint-Jacobian
  rank. (G10)
- **E-3** `I_eff`, `dI_eff/dθ`, `G`, `S`, `r_i` MUST be analytic via first- and
  second-order implicit differentiation. No `np.gradient` on table quantities.
- **E-4** The validity gate of §3 MUST be applied by the interpreter to every
  candidate of every topology. (G2, G7, G8)
- **E-5** `table.cond` MUST carry a measured conditioning value; a constant is a
  test failure. (G2)
- **E-6** Residual and Jacobian assembly MUST be vectorised over the population
  axis. (§2.5)
- **E-7** Energy MUST be per-actuator, efficiency-weighted, including hold power.
  (G6)
- **E-8** Rotary drives MUST carry a gear-ratio gene and reflected inertia
  `G_r²·J_rotor`; until then they MUST be excluded from `auto` seeding. (G1)
- **E-9** Zero-capacity and slack actuators MUST NOT constrain `omega_lim`, and
  secondary capacity MUST decode to zero rate at gene 0. (G5)
- **E-10** `add_dyad` / `remove_dyad` / `split_link` MUST be implemented and MUST
  preserve mobility by construction. (G4)
- **E-11** Every structural operator MUST be behaviour-preserving at birth,
  verified over ≥ 20 seeds — not one deduplicated draw. (G5)
- **E-12** `--limb` + `--engine graph` MUST either work or be rejected at
  argument parsing. (G3)
- **E-13** Spring free length/angle MUST decode against fixed physical bounds.
- **E-14** The robustness screen MUST be a hard gate.
- **E-15** Hold-out MUST use interleaved repetitions, and the grid MUST cover
  both windows.

**Visualization**

- **V-1** Each run MUST emit `scene.json` + `scene.glb`; no consumer may
  recompute a pose. (G3, G9)
- **V-2** `printable.solid_model` MUST derive geometry from the scene. (G9)
- **V-3** Anatomical solids MUST be generated from measured segment lengths,
  with named landmarks.
- **V-4** Every render MUST carry the representative-geometry caveat (R-V1) and
  annotate the fixed-axis approximation (R-V2).
- **V-5** The run page MUST provide an interactive 3D viewer with keyboard
  control and a non-WebGL fallback.
- **V-6** three.js MUST be vendored, not fetched from a CDN.
- **V-7** Offscreen rendering MUST degrade through the §10 chain and MUST NOT
  fail a run.
- **V-8** Every figure MUST pass `figures.check` (§11.1); the check MUST run in
  CI over every figure the pipeline can emit.
- **V-9** All 17 form labels MUST gain `for=`, and the viewer MUST expose an
  `aria-live` state summary.

---

## 13. Validation

**Unit**

- Generic interpreter vs the retained hand-written `four_bar`: tables must agree
  to 1e-10 across the ROM. Same for `direct_linear` vs `mechanism.py`, which must
  still hold **bit-for-bit** (the Phase 1 contract is not to be regressed).
- Analytic `x′`, `x″` vs central differences (targets: 1e-9, 1e-8).
- Mobility: Kutzbach and Jacobian rank agree on a fixture set including a
  deliberately overconstrained linkage that Kutzbach alone passes.
- Validity gate: fixtures for each of V1–V9, including the measured G7 genome
  (`|r| = 1.19e-10 m` at θ = 94.8°) and a link-crossing four-bar.
- Behaviour preservation: every operator, ≥ 20 seeds, `< 1e-9`.
- Energy: opposed pair must cost strictly more than a single actuator delivering
  the same trajectory (today both read 10.26 J).
- `figures.check` over every figure factory.

**Integration**

- Full pipeline per topology, both engines, single-joint and limb, asserting
  scene, spec, STL, viewer payload and all figures.
- **Non-regression against the legacy engine:** the graph engine's
  `direct_linear` winner MUST satisfy every constraint the legacy engine
  enforces. Today it fails two — stroke ratio 2.45 (limit 1.8) and packaging
  margin −58.9 mm. That comparison becomes a permanent test.

**Physical plausibility review**

Unchanged in spirit and still mandatory: the top ten designs of each phase gate
reviewed by someone willing to build one. With anatomical renders this gets
easier and more honest — a mechanism that intersects the humerus is now visible
rather than latent in a number.

---

## 14. Phasing and acceptance

| Phase | Content | Accept when |
|---|---|---|
| **R1** | Stop the bleeding: E-12 (G3), E-8 seeding exclusion, `report.md` text, `actuator_requirements` masking, quota starvation, spring decode | `--limb --engine graph` no longer crashes; `auto` runs no longer trivially select rotary |
| **R2** | Universal validity gate on the existing three topologies (E-4, E-5) + non-regression test | graph `direct_linear` winner satisfies every legacy constraint; the G7 and G8 fixtures are rejected |
| **R3** | Generic interpreter (E-1, E-2, E-3, E-6) with the hand-written classes as oracles | agreement to 1e-10; batched budget met (§2.5) |
| **R4** | Fitness corrections + structural operators (E-7, E-9, E-10, E-11, E-13, E-14, E-15) | opposed pair costs more than single; ≥ 20-seed preservation passes; search discovers a non-seeded topology that wins on total cost |
| **R5** | Scene export + parametric anatomy (V-1, V-2, V-3, V-4) | printed parts and simulated mechanism provably share geometry; limb GIFs come from the scene |
| **R6** | 3D viewer + offscreen renderer (V-5, V-6, V-7) | viewer works with keyboard only; headless CLI still produces images with WebGL absent |
| **R7** | 2D diagram standard (V-8, V-9) | `figures.check` passes over every figure; zero unlabelled controls |

R1 and R2 are small and should land before anything else — R2 in particular,
because until it does the graph engine is *worse* than the legacy engine at the
legacy engine's own topology, and every subsequent result is measured against a
broken baseline.

---

## 15. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Generic interpreter is too slow | High | Measured 203× batching speedup (§2.5); assembly vectorisation is a hard requirement, and R3 does not pass without the budget |
| Newton non-convergence on exotic graphs | Medium | Warm start, analytic Jacobian, convergence failure is a V1 rejection — never an exception |
| Generic solver regresses the verified `direct_linear` bit-for-bit match | High | Hand-written classes retained as oracles; the Phase 1 test stays as-is |
| Anatomical renders imply clinical validity | **High** | R-V1/R-V2/R-V3 are acceptance-blocking, not cosmetic. This is the risk that matters most outside the codebase |
| Offscreen 3D is fragile headless | Medium | Three-tier fallback (§10); rendering never fails a run |
| Structural search still collapses to one topology | Medium | Dyad primitive makes growth safe; speciation retained; R4 acceptance explicitly tests for discovery of a non-seeded winner |
| Scene format churn | Low | `schema_version` from day one; viewer and renderer both version-gated |

---

## 16. Out of scope

Unchanged from the prior spec — 3D mechanisms, mobility > 1, multi-DOF
anatomical joints, inter-joint coupling, control co-design, compliant mechanisms,
vendor catalogue matching, real-time simulation — plus:

- **Patient-specific anatomy.** No imaging, no registration, no subject-specific
  bone geometry. Segment lengths are the only measured input, and §8.3 requires
  saying so.
- **Muscle or soft-tissue mechanics.** The soft-tissue envelope is a visual
  boundary, not a model.
- **Migrating joint axes.** The revolute stays fixed; R-V2 requires annotating
  that rather than fixing it.

---

## 17. Open questions

1. Does V3 (`min|r|`) fully subsume the transmission-angle check for six-bar and
   cam topologies, or is a per-dyad angle still needed as a gate rather than a
   diagnostic?
2. What `MIN_MOMENT_ARM` equivalent applies to a rotary drive — output torque
   ripple? backlash-limited resolution?
3. Should `add_dyad` be allowed to attach to the OUTPUT body directly, or only
   to couplers? Attaching to OUTPUT changes the driven kinematics most
   aggressively — highest reward, highest lethality.
4. Interleaved hold-out: split by repetition (needs cycle detection) or by fixed
   interleaved windows (simpler, slightly leaky)?
5. GLB per frame is wasteful for long recordings — is a fixed mesh set plus a
   pose track sufficient for every topology, including cams whose profile is
   geometry rather than pose?
6. Do the anatomical solids need a left/right mirror check against the measured
   side, and how is that validated without imaging?
