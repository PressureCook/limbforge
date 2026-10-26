# Code review round 2 — graph engine, physics fixes, web UI

Reviewed at 2026-08-12, against the previous review (`REVIEW.md`) and
`SPEC-open-topology-evolution.md`. ~1,750 new lines of engine
(`topology.py`, `topologies.py`, `reduced_sim.py`, `graph_evolve.py`,
`runner_graph.py`, `graph_viz.py`) plus 560 lines of tests and edits across 15
existing files. All 35 tests pass. Every number quoted below was measured by
running the code.

---

## Verdict

**Phase 0 (physics prerequisites) and Phase 1 (reduction infrastructure) are
done, and done well.** The anthropometry rework is genuinely careful — Winter's
COM and radius-of-gyration values are quoted as self-consistent proximal pairs,
`theta_neutral` is the right abstraction for the gravity convention, and
payload now enters properly as (mass, distance). `direct_linear` reproduces
`mechanism.py` bit-for-bit. That was the phase most likely to go quietly wrong
and it didn't.

**Phase 2's validity gate and Phase 3's structural search are the problem.** Two
of the three topologies ship `valid=ones, cond=ones` — no feasibility gate at
all — so the open-topology path silently dropped every geometric constraint the
legacy engine enforced, and the fitness function's singularity penalty is
identically zero. Meanwhile the mutation operators cannot change topology, so
what exists is a three-entry curated library with a graph-shaped genome, not the
open search that was specified.

The two combine into the headline result: **the search converges on
`rotary_direct` — the topology with the least-modelled physics — in both of my
runs, by generation 10, and then flatlines.** It wins because it is cheapest to
model, not because it is the best design.

---

## Round-1 findings: closed

Verified fixed. Good work — this is nearly the whole list.

| ID | Fix | Verified |
|---|---|---|
| **C1** gravity convention | `theta_neutral` per joint; `-G·lever·sin(θ−θ_n)` | release-from-rest test now passes for all five joints |
| **C2** power sign | `P_mech = -F_tr * v_tr` | `test_power_sign_convention` |
| **C3** anthropometry | `L_mass` + `MASS_END_LANDMARK`; Winter proximal `com_frac`/`k_frac` pairs | shoulder 0.530/0.645, hip 0.447/0.560, forearm+hand 0.682/0.827 — all correct against Winter |
| **C4** photo timestamps | original indices preserved through subsampling | `test_photo_timestamps_survive_subsampling` |
| **C5** unbounded video RAM | hard cap inside the read loop + `_downscale` to 1080 px | `test_video_capped_even_if_frame_count_unreadable` |
| **M1** payload inertia | `payload_dist` used for both `m·d` and `m·d²` | `test_payload_adds_demand_at_distance` |
| **M2** inertia formula | `m·(k_frac·L)²` replaces `m·L²/3` | — |
| **M5** clipped `F_peak` | `F_demand_peak` recorded and surfaced ("134% of capacity demanded") | run log |
| **M3** 45° drawings | `th_mid` parameter threaded through | — |
| **M4** `index.json` race | `_INDEX_LOCK` | — |
| **M7** single-joint chips | `packaging!` chip now outside the limb loop | run page |
| **M9** poll ceiling | `fails + 1` counter | — |
| **M11** hidden spec heading | `h2` renders | measured `display: block` |
| **H1a** vacuous ROM check | raises when required joints absent | `test_missing_required_joint_fails` |
| **H1b** skipped checks | `capture_verified` posted and accepted | `app.py:123` |
| **H2** camera teardown | `stopCamera()` clears timer and stops tracks | — |
| **H3** side override | gated to record mode *and* `side == "auto"` | — |
| **H4** `..` traversal | `^[A-Za-z0-9_-][A-Za-z0-9._-]*$` | — |
| low | `\| tojson`, `revokeObjectURL`, `aria-live` (4), `pyproject.toml`, `.gitignore`, pytest suite | — |

Still open from round 1, all minor: **0 of 17 `<label>` elements have `for=`**
(the `aria-live` regions were added but the labels themselves are still
unassociated); M6 packaging-wall; M8 `#coachTip` contention; M10 hold-power
model.

---

## Critical

### G1 — The search converges on the topology with the least-modelled physics

`topologies.py` `rotary_direct` sets `r = +1` identically, `I_eff = ctx.inertia`,
`dI_dtheta = 0`, `valid = ones`, `cond = ones`. There is no gear-ratio gene, no
reflected rotor or gearbox inertia, no moment-arm constraint, no stroke, no
packaging envelope, no singularity — the simulator hands it perfect torque
transmission at every angle for free.

Two independent runs on the elbow demo (pop 120/gen 60 and pop 60/gen 30, auto
seeding at 60/25/15% direct_linear/four_bar/rotary):

```
gen   0  best 5.24  species 3  best-structure rotary_direct/1act/0spr
gen  10  best 4.73  species 10 best-structure rotary_direct/1act/1spr
gen  59  best 4.66  species 9  best-structure rotary_direct/1act/1spr
```

Converged by generation 10, then 4.67 → 4.66 across the remaining 50
generations. The shipped design is `28-32 mm BLDC class @ 160:1`. At that
ratio the reflected rotor inertia of a 30 mm BLDC (J ≈ 1e-6 kg·m²) is
**160² × 1e-6 = 0.026 kg·m², about 29% of the limb's own 0.088 kg·m²** — the
single largest dynamic term in the drivetrain, and the model does not contain
it. `rotary_direct` is therefore credited with tracking it cannot achieve, and
the topology comparison is decided by that omission.

Fix: give `rotary_direct` a gear-ratio gene, put `G_r²·J_rotor` into `I_eff`,
and give it a real validity gate (peak output torque, backlash, envelope). Until
then it should be excluded from `auto` seeding, because it will win every time.

### G2 — The graph path silently dropped every geometric feasibility constraint

`evolve.py:55-61` enforces, as hard penalties: `stroke_ratio ≤ 1.8`,
`L_ret ≥ 40 mm`, `r_min ≥ 8 mm`, `stroke ≥ 8 mm`, static holdability, and
packaging (`L_ret ≥ stroke + 28 mm`). `graph_evolve._cost_terms` has **no
equivalent for any of them**. Its only geometric term is

```python
sing = float(1.0 / max(np.max(table.cond[in_rom]), 1.0))
... + W_SING * max(0.0, 0.01 - sing) / 0.01
```

which reads `table.cond` — hardcoded to `1.0` by `direct_linear` and
`rotary_direct`, making `sing = 1.0` and the penalty identically zero for both.

Measured, same problem, same seed, 120×60:

| | graph engine (`direct_linear`) | legacy engine | legacy limit |
|---|---|---|---|
| min moment arm | 22.3 mm | 13.3 mm | ≥ 8 mm |
| **stroke ratio** | **2.45** | 1.61 | ≤ 1.8 |
| retracted length | 68.6 mm | 101.6 mm | ≥ 40 mm |
| **packaging margin** | **−58.9 mm** | +12.0 mm | ≥ 0 |

`table.valid.all() == True` — nothing was rejected. The winning design needs
127.5 mm of actuator body inside a 68.6 mm retracted length. It cannot be built.

Across 2,000 random `direct_linear` genomes, all fully "valid": 56% below the
8 mm moment-arm floor, 27.7% over the stroke-ratio limit, 1.8% under the 40 mm
minimum retracted length.

Fix: `direct_linear.build_table` must populate `valid` from the same constraints
`evolve._design_penalties` applies, and `cond` must carry a real conditioning
measure (for this topology, `|r|`) rather than a fabricated 1.0.

### G3 — `--limb <arm|leg> --engine graph` crashes

Reproduced with `run_pipeline.py --demo --limb arm --engine graph`:

```
=== Full-limb simulation and report ===
  limb/usage animations use the legacy engine's mechanism objects; ...
ERROR: list index out of range
  File "runner.py", line 417, in _run_limb
    viz.make_limb_gif(cfg.limb, side, series_map, finals, mechs, ctxs, ...)
```

`runner_graph.process_joint_graph` returns `"mech": None`, `_run_limb` builds
`mechs = {j: r["mech"]}` from it, and `make_limb_gif` dereferences it. The tee
message implies graceful degradation; there is none. `run_usage_scenario` on the
next line would fail the same way.

Fix: either give `runner_graph` a `mech`-shaped adapter for the drawing code, or
route `make_limb_gif`/`run_usage_scenario` through `graph_viz` for graph runs —
and until then reject the `--limb` + `--engine graph` combination at argument
parsing rather than 60 seconds into the run.

### G4 — Structural search cannot change topology

`_mutate_structure` offers exactly four operators: `add_spring`,
`add_antagonist` (only when `topology == "direct_linear"` and there is one
actuator), `remove_spring`, `remove_actuator`. There is no `split_link`, no
`add_via_point`, no `retype_actuator`, no `add_coupler_loop`, no `simplify`, and
nothing that alters `genome.topology`.

The three topologies in `topologies.py` are hand-written closed-form kinematics
(`four_bar` hardcodes `l1`, `l2`, `b_r`, `P0`, `A`), reachable only through
seeding and immigration. `genome.bodies` and `genome.joints` exist in the
dataclasses but no topology reads them generically.

That is the **curated topology library** — the option explicitly not chosen —
wearing a graph-shaped genome. It's a reasonable place to be at this stage, but
it should be named as such: what ships today cannot discover a mechanism nobody
wrote down, which was the point. The observable consequence is the flat
convergence in G1: after generation 10 the only structural move available is
adding or removing a spring.

---

## High

### G5 — `add_antagonist` is not behaviour-preserving, and its test passes by luck

`topologies._decode_secondary_capacity` returns
`(gF**2 * 800.0, 0.02 + 0.33*gv)` — force decodes to 0 at gene 0, but **speed
decodes to a 0.02 m/s floor**. `reduced_sim` then computes

```python
omega_lim = np.min(batch.rate_max / np.maximum(np.abs(r), 1e-4), axis=1)
```

a *min over all actuators*, so a zero-force actuator still throttles the joint.
Measured over 40 seeds: **8 of 40 `add_antagonist` mutations changed the
trajectory, worst case 63.85°.**

`test_structural_mutations_behaviour_preserving_at_birth` asserts `< 1e-9` but
keys `seen` on `(n_actuators, n_springs)` and `continue`s on duplicates, so it
only ever tests the **first** antagonist draw at seed 0 — which happens to be
benign. The test is real but its coverage is one sample.

This is the same failure mode the spec warned about: a lethal structural
mutation means the search can never cross into antagonist topologies.

Fix: floor the secondary rate at 0 alongside force; exclude zero-capacity
actuators from the `omega_lim` min; and make the test loop over ≥20 seeds
instead of deduplicating by shape.

### G6 — The energy fitness term cannot see antagonistic co-contraction

```python
energy += np.maximum(-(u * (-r * omega[:, None])).sum(axis=1), 0.0) * dt
```

This reduces to `max(τ_net · ω, 0)` — **net** joint power. Internal work between
opposed actuators cancels before the sum. Measured with a hand-built opposed
pair driving the demo trajectory:

```
single actuator            10.26 J
opposed pair (both working) 10.26 J   -> no extra cost seen
```

Energy was specified as the primary discriminator for multi-actuator topologies,
and co-contraction waste is the thing that should disqualify a bad antagonistic
design. It is invisible.

Separately, this is *mechanical joint work*, not the C2-corrected electrical
integral in `actuator_design._energy_analysis`. Drivetrain efficiency, holding
power and idle draw are all absent, so a self-locking leadscrew and a
backdrivable ballscrew are indistinguishable to fitness.

Fix: `Σ_i max(u_i · r_i · ω, 0)` per actuator, and scale each by that
actuator's `eta_tot` so efficiency enters the objective.

### G7 — `four_bar` checks the wrong transmission angle

The code measures the **crank–coupler** angle; the angle that governs force
delivery in a four-bar is **coupler–output**. Measured on a genome with
`valid.all() == True`: at θ = 94.8° the checked angle is a healthy 119.4°, the
unchecked one is **3.5°**, `|r| = 1.19e-10 m`, and holding the 12.5 N·m gravity
load there would need **1.0e11 N**.

Across 4,000 random genomes, of the 710 valid everywhere, **66% have `r` change
sign inside the ROM** (an exact dead point) and 86% dip below the legacy 8 mm
moment-arm floor. Median `min|r|` among fully-valid designs: **0.1 mm**.

### G8 — No self-intersection check anywhere; the envelope check covers 2 of 5 points

`four_bar` envelope-tests only `C` and `com2`; `D`, `B` and the actuator line
`A–D` are untested, and no topology checks whether a link passes through another
link or through a bone. Measured: **92.1% of fully-valid `four_bar` designs**
have the crank, coupler or actuator line crossing a bone somewhere in the ROM;
**74.5% of `direct_linear`** genomes have the actuator line passing through a
bone. The fully-valid `four_bar` minimum actuator length reaches 4.5 mm, against
a 28 mm dead length.

### G9 — `solid_model` ships geometry unrelated to the simulated mechanism

`direct_linear` and `rotary_direct` emit fixed-size stubs:

```python
Barrel(od=20 * scale, bore=8 * scale, length=60 * scale),
Rod(d=6 * scale, length=50 * scale)]
```

`printable.from_parts` turns these into real STL cylinders and ships them. From
the actual graph run I executed:

```
part_plate_1.stl    78.0 x 18.0 x  6.0 mm     (ctx.Ld = 270 mm)
part_barrel_1.stl   35.9 x 35.9 x 21.5 mm
part_pin_1.stl       8.0 x  8.0 x 25.0 mm
```

A 78 mm plate for a 270 mm forearm, plus two generic blocks. 97% of random
`direct_linear` genomes need more than the ~110 mm the stub barrel+rod can
reach; the median needs 190 mm retracted / 294 mm extended.

This is the compatibility requirement (§1.3 / R-17) failing in the way the spec
was written to prevent: the chain runs to completion and produces files, so
nothing errors — but the printed assembly is not the simulated mechanism.
`four_bar`'s plates *are* at true dimensions; only its barrel/rod are stubs.

### G10 — `mobility()` is `return 1` in all three topologies, and the gate asserts `== 1`

No Kutzbach count, no Jacobian rank. `topology._smoke_test` does
`assert topo.mobility(genome) == 1`, so the registration-time mobility check can
never fail — including for a future topology whose DOF is genuinely wrong, which
is precisely what R-5 was for.

---

## Medium

- **Spring free angle is decoded from the θ-grid endpoints**
  (`free = theta_grid[0] + gene*(theta_grid[-1]-theta_grid[0])`), and the
  evolution grid (fit-window ROM, 128 pts) differs from
  `simulate_graph_final`'s (full-recording ROM, 256 pts). Measured with
  `k=0.6, free=0.5`: free angle 37.5° during evolution vs 57.5° in the final
  run, and spring torque at θ=40° goes **−0.78 → +5.49 N·m, a sign flip**. The
  reported design is not the design that was evolved. Decode against fixed
  physical bounds.

- **The hold-out split measures the wrong thing.** `fit_mask` takes the first
  70% of the recording, so for the demo (three curls then a static hold from
  t=6.9 s) the hold-out is dominated by the hold. A generalisation metric
  computed on a stationary segment is close to meaningless. The spec suggested
  alternate repetitions; that would be better. Related: `th_lo/th_hi` come from
  the fit window only, so the transmission table is never built over the
  hold-out's ROM.

- **`actuator_requirements` ignores `table.valid`.** At non-assembling grid
  points `four_bar` freezes the Newton iterate, and that leftover `q` feeds
  `stroke_mm`, `retracted_mm` and the packaging check. Measured: reported stroke
  22.2 mm vs true 13.5 mm (64% over). Conservative direction, but wrong.

- **The robustness screen is not a hard gate.** §8.3 said reject any design that
  goes invalid under perturbation; `evolve_graph` tries five candidates and, if
  all fail, ships `best_genome` anyway with only a log line.

- **`I_eff`, `G` and `dI` are finite-differenced** (`np.gradient`) while `r` and
  `dphi` are analytic and available in closed form 20 lines earlier. Currently
  latent — link mass is ~0.4% of `I_eff`, so the measured error is ~1e-9 — but
  it makes the table grid-density dependent and it straddles the discontinuity
  at frozen points.

- **Branch-flip guard invalidates only the flipping sample.** It compares
  against `phi[:, i-1]`, which is itself already flipped, so subsequent points
  compare small and are marked valid on the wrong assembly. The 0.6 rad
  threshold is absolute rather than scaled by `|dphi|·Δθ`, so it does not track
  grid density (64 / 128 / 256 across smoke test / evolution / final). I could
  not produce a genome that triggers it — reporting as latent.

- **`u_demand_peak` is computed under a different allocation than the actual
  one** (`allocate_greedy` with capacities forced to 1e9, which changes the
  weights). Exact for one actuator, approximate for more — and it feeds the
  `W_SAT` fitness term. It also costs a second full allocation every timestep.

- **A slack tendon still throttles joint speed**, because `omega_lim` takes the
  min over all actuators including unilateral ones that are not currently
  pulling.

- **Offspring quotas starve the tail species.** Species elites are appended
  outside the `pop_size - n_imm` break, then `new_pop[:pop_size]` truncates.
  Simulated at 18 species: children per species
  `[31,15,10,7,6,5,4,3,3,3,2,2,2,2,0,0,0,0]` — the last four get nothing but
  their elite, and immigrants drop from the intended 9 to 6.

- **The `½ dI/dθ · θ̇²` term is not in the controller feed-forward** while
  gravity is. Legitimate as a design choice, but the asymmetry biases the search
  toward topologies with small `dI/dθ`, which may not be intended.

- **`report.md` still says** "gravity assumes the proximal segment vertical
  (standing capture)" — stale since the C1 fix replaced `gravity_mode` with
  `theta_neutral`.

---

## Low

- **0 of 17 `<label>` elements have `for=`** (measured in the browser). The four
  `aria-live` regions are a real improvement, but every select and number input
  is still an unlabelled control to a screen reader, and clicking the visible
  text does not focus the field.
- The disabled "Start guided recording" button gives no reason — no `title`, no
  adjacent hint. On a fresh load the movement select defaults to "free movement"
  (`value=""`), so `currentPhases()` is null and the button stays disabled even
  after enabling the camera and picking a joint. Correct behaviour, invisible
  rationale.
- `GET /favicon.ico → 404` on every page load.
- Carried over: M6 (packaging constraint is a hard wall, so margin is always
  ≈0 and "FITS" carries no information), M8 (`#coachTip` written by two timers),
  M10 (hold-power model drops gear efficiency, uses a magic 0.8).

---

## What's genuinely good

- **The anthropometry rework.** Winter's proximal COM/radius-of-gyration pairs
  are quoted correctly and self-consistently (forearm+hand 0.682/0.827,
  shank+foot 0.606/0.735, whole arm 0.530/0.645, whole leg 0.447/0.560), with
  `MASS_END_LANDMARK` measuring the right length. This is the fix I'd have been
  least surprised to see done half-way, and it wasn't.
- **`theta_neutral`** is the right abstraction — it makes the gravity convention
  a per-joint datum instead of a mode flag, and the release-from-rest test is
  exactly the acceptance test to guard it.
- **`direct_linear` reproduces `mechanism.py` bit-for-bit** (`np.array_equal`,
  not `allclose`) including the payload columns. The Phase 1 contract is met.
- **`four_bar`'s core maths is right**: `r` matches a 4th-order central
  difference of `q` to 5e-12, `I_eff` matches an independent kinetic-energy
  computation to 1e-8 with parallel-axis handled correctly, `G` matches
  `-Σ m_k g dy_k/dθ` to 4e-7 N·m on 12 N·m, and the implicit-function derivative
  and analytic Jacobian are correct. The problem is the gate around it, not the
  kinematics.
- **`allocate_qp`** — solving the box-constrained QP through the KKT
  multiplier with a vectorised bisection is a neat piece of work, and it is
  correct. Measured greedy-vs-QP gap on the winner: **1.2e-13 degrees**.
- **The test suite is real.** `test_physics.py` is precisely the acceptance set
  the spec asked for, and `test_capture.py` covers exactly the two bugs that had
  no coverage before.
- **The site.** Clean, legible, no broken images, no JS errors beyond the
  favicon, and nearly every round-1 UI defect closed. The run page now surfaces
  topology, hold-out RMSE and the packaging chip.

---

## Suggested order

1. **G3** — one crash, reproducible, and it makes half the CLI surface a trap.
2. **G2** — port `_design_penalties` into `direct_linear.build_table`'s `valid`
   mask and give `cond` a real value. Without this the graph engine is
   *worse* than the legacy engine at its own seeded topology, which undermines
   the whole migration.
3. **G1** — add the gear-ratio gene and reflected inertia to `rotary_direct`, or
   drop it from `auto` seeding until you have. Right now every `auto` run is
   really a rotary run.
4. **G7 + G8** — the four_bar gate: correct the transmission angle, add
   self-intersection, extend the envelope test to all points. Then re-run and
   see whether four_bar is still competitive when it has to be buildable.
5. **G5** and **G6** — the two reasons structural search can't currently find
   anything interesting: the antagonist mutation is lethal and the fitness can't
   see why an antagonist would be bad.
6. **G9** — make `solid_model` build from the evolved dimensions. Until then the
   STL output for two of three topologies is decorative.
7. **G4** — decide and document whether open topology mutation is still the
   goal. If yes, the next step is a generic body/joint interpreter that reads
   `genome.bodies`/`genome.joints` instead of three hardcoded kinematic
   functions; if no, update the spec and the README so the claim matches what
   ships.
