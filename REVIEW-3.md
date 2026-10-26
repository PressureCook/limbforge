# Code review round 3 — generic interpreter, universal gate, rectification

Reviewed 2026-08-12 against `SPEC-2-rectification-and-visualization.md` and
`REVIEW-2.md`. New: `interpreter.py` (290 lines), `gate.py` (134),
`tests/test_interpreter.py`, `tests/test_rectify.py`. Changed: `topologies.py`
(+164), `graph_evolve.py` (+88), `reduced_sim.py` (+43), `runner.py`,
`run_pipeline.py`, `app.py`. Every number below was measured by running the code.

Part II (visualization, R5–R7) has not been started — `visualize.py`,
`graph_viz.py`, `printable.py` and the templates are untouched and there is no
`static/`. That is the sequencing the spec asked for, not a gap.

---

## Verdict by phase

| Phase | Status |
|---|---|
| **R1** — stop the bleeding | **Done.** `--limb --engine graph` rejects at argument parsing with an accurate message; rotary excluded from `auto` seeding with an honest comment; `auto` now selects `direct_linear` with a real convergence curve (8.19 → 5.49 over 25 generations) instead of round 2's flat rotary 4.67 → 4.66. |
| **R2** — universal validity gate | **Substantially done, one hole.** The headline non-regression now passes. But V8/V9 sit behind `hasattr(topo, "pose_points")` and two of three topologies don't implement it (H2). |
| **R3** — generic interpreter | **Structurally done; misses its performance budget by ~56×, and its own acceptance test fails** (H1). The maths is correct. |
| **R4** — fitness + structural operators | **Fitness partial; operators 0 of 7** (H3, H4, H5). |
| **R5–R7** — visualization | Not started. |

### R2's acceptance criterion is met

This was the most important single thing in the spec, and it landed. Graph
engine, `direct_linear`, measured on the elbow demo:

| | round 2 | **now** | legacy limit |
|---|---|---|---|
| min moment arm | 22.3 mm | **24.4 mm** | ≥ 8 mm |
| stroke ratio | **2.45 ✗** | **1.66 ✓** | ≤ 1.8 |
| retracted length | 68.6 mm | **153.0 mm** | ≥ 40 mm |
| packaging margin | **−58.9 mm ✗** | **+23.4 mm ✓** | ≥ 0 |

The graph engine is no longer worse than the legacy engine at the legacy
engine's own topology. Everything downstream is now measured against a sound
baseline.

---

## What is genuinely right

- **The interpreter's maths.** I verified the parts most likely to be subtly
  wrong. `_Poses.point`'s second derivative `w″ = p″ − R·loc·(φ′)² + R′·loc·φ″`
  is correct. `_second_rhs` correctly includes the OUTPUT body and correctly
  drops GROUND. The second-order implicit differentiation is implemented exactly
  as specified — no `np.gradient` anywhere on table quantities.
- **The gravity encoding, which is the subtlest thing here and is right.**
  `g_hat = [cos θ_n, −sin θ_n]` is not an arbitrary rotation: it is precisely the
  true "up" direction implied by requiring the output COM to hang lowest at
  θ = θ_n, and it reproduces the legacy `−mg·com·sin(θ − θ_n)` exactly for every
  joint. That means it generalises correctly to coupler bodies, which the legacy
  scalar formula could not. Easy to get wrong; not got wrong.
- **Assembly is genuinely vectorised over the population axis (E-6).** The only
  Python loops are over the joint list, whose length is a structural constant of
  the species. `_sanitize` isolating diverged candidates so they cannot perturb
  their batch-mates' arithmetic is a nice touch.
- **E-12, E-13, E-14 complete.** Limb rejection is clean; spring free angle now
  decodes against fixed physical bounds (verified identical to 1.1e-16 across
  two different θ grids); the robustness screen raises and propagates to a
  `status: failed` run.

---

## Critical

### H1 — Performance misses the budget by ~56×, and the acceptance test fails

Measured, `four_bar.build_table_batch`, P = 120, 128 grid points:

```
1.83 s  =  15.27 ms/candidate      (spec §2.5 budget: 0.275 ms/candidate)
```

`tests/test_interpreter.py::test_species_batched_budget` asserts
`< 2.0 s for 120 candidates` — a threshold already **60× looser** than the spec
it cites — and **fails**:

```
E   AssertionError: batched precompute took 2.08s for 120 candidates
E   assert 2.077962264999769 < 2.0
```

End-to-end: an `auto` run at pop 60 × gen 25 took **108 s**, where round 2 did
pop 120 × gen 60 in 100 s — roughly **4× slower per unit of work**. The test
suite no longer completes: **29 of 48 tests after 25 minutes**, still running.

cProfile (`tottime`) points at three cheap fixes:

```
1664 calls  0.377s  interpreter.py:127(_assemble)          [1.454s cumulative of 1.83s]
60682 calls 0.365s  numpy .../shape_base.py:378(stack)
1664 calls  0.129s  numpy .../linalg.py:2355(det)
```

- **(a) The Newton loop never exits early.** `for _ in range(NEWTON_ITERS)` with
  `NEWTON_ITERS = 12` runs all twelve every grid point. 1664 = 128 × 13 confirms
  it. Warm-started Newton converges in 2–4 — my own prototype measured 385 total
  iterations across a 128-point sweep. **~3–4×.**
- **(b) `_rot`/`_drot` build 2×2 matrices through nested `np.stack`** on every
  call — 60,682 calls, 0.54 s. Direct `np.empty` + slice assignment, or hoisting
  `cos`/`sin`, removes most of it.
- **(c) `_sanitize` computes a full determinant every iteration** to detect
  singularity, when the solve already signals it.

**One correction I owe you:** the spec's 0.275 ms/candidate figure came from a
solve-only microbenchmark that explicitly excluded assembly. Assembly here *is*
correctly vectorised — E-6 is satisfied — so that number was wrong, not the
approach. After (a)–(c) a realistic figure is ~1.5–2.5 ms/candidate. **Amend
§2.5 to that rather than chasing the code toward an unreachable number**, and
retune the test threshold to the amended budget instead of leaving it at a
value 60× loose that still fails.

### H2 — V8/V9 are behind `hasattr`, so two of three topologies skip them

```python
if hasattr(topo, "pose_points"):
```

`direct_linear` and `rotary_direct` do not implement `pose_points`. Measured
over 600 random `direct_linear` genomes:

```
445 (74%) have the actuator line crossing a bone
5 of those PASSED the gate
```

Since the gate's overall pass rate on `direct_linear` is ~2.8% (≈17 of 600),
**roughly 30% of gate-passing direct_linear designs intersect a bone.**

This is the same opt-out `valid=np.ones` was, relocated from a data default to a
duck-typed hook. The spec's wording was "applied by the ENGINE to every
candidate of every topology, so feasibility cannot be skipped by a topology
class" — `hasattr` is precisely that skip.

Fix: add `pose_points` to `REQUIRED_METHODS` in `topology.py:198` so
registration fails without it, and have it return the bone segments too, not
only mechanism links.

### H3 — E-10/E-11: 0 of 7 structural operators; topology is still fixed at seeding

`_mutate_structure` is **byte-identical** to the previous snapshot — `diff`
produces no hunk for that region. `grep` for `add_dyad|remove_dyad|split_link|
add_via_point|retype_actuator|move_attachment|simplify` across the package and
tests returns nothing.

More consequential: **the interpreter never reads `genome.bodies` or
`genome.joints`.** Each topology class hand-writes its own `graph_spec_batch`
from decoded scalar genes (`topologies.py:158, 344, 599`), and `Genome.bodies` /
`Genome.joints` stay empty. So no mutation can add a coupler body, and R4's
acceptance — "the search discovers a non-seeded topology that wins on total
cost" — is structurally unreachable.

Worth stating plainly because it is easy to read the interpreter as having
closed G4: **it unifies the solver, not the representation.** That is real
progress and it is the necessary precondition for G4, but G4 is still open.
`add_antagonist` is also still guarded by `topology == "direct_linear"`, so
`four_bar` — 35% of the `auto` seed — has no structural operator at all beyond
springs.

---

## High

### H4 — The energy rewrite is arithmetically inert under both shipped allocators

```python
P_i = np.maximum(u * r * omega[:, None], 0.0) / eta_vec[None, :]
```

`allocate_greedy` gives every `r_i·u_i` the sign of `τ`; `allocate_qp` gives
every `r_i·u_i` the sign of `λ`. In both cases all per-actuator powers share a
sign, so `Σ max(x_i, 0) ≡ max(Σ x_i, 0)` — the rewrite is a constant `1/η`
rescale of the old net-torque formula. Measured, identical trajectory in every
row:

| configuration | round 2 | now |
|---|---|---|
| single linear | 3.5575 J | **6.4682 J** |
| opposed pair, greedy | 3.5575 J | **6.4682 J** |
| opposed pair, QP | 3.5575 J | **6.4682 J** |
| two identical linear | 3.5575 J | **6.4682 J** |

6.4682 = 3.5575 / 0.55 exactly; the hold term contributed 0.000 J. §13's
acceptance ("opposed pair must cost strictly more than a single actuator
delivering the same trajectory") is still unmet. The test that claims to cover
it monkeypatches in a co-contracting allocator the shipped code never produces —
and still passes with `ETA_CHEAP` set to all 1.0 and `HOLD_W_PER_EFFORT` to all
0.0, so it covers neither of E-7's other two clauses.

Also: η is keyed on actuator *kind* only (`{"linear": 0.55, …}`), so within a
species it is a constant multiplier and cannot reorder anything.
`actuator_design` already carries `ETA_BALLSCREW`, `kind`, and `self_locking` —
none of it is reachable, so §4.1's "self-locking is finally distinguishable from
backdrivable" is not achieved, and the required cheap-vs-exact reconciliation
report does not exist.

### H5 — The zero-capacity exclusion is a knife-edge at 1 µN

```python
active = (batch.u_max > 1e-6) & (~batch.unilateral[None, :] | (np.abs(u) > 1e-6))
```

Secondary force decodes as `gF²·800`, so gene `F = 1e-4` → `u_max = 8 µN` —
physically zero, but above the threshold, so the rate limit is reinstated:

```
F_gene=0.0      u_max=0.0 N      traj change  0.000 deg   rmse 25.216 -> 25.216
F_gene=0.0001   u_max=8.0e-06 N  traj change 79.024 deg   rmse 25.216 -> 57.163
```

So 40/40 `add_antagonist` births are now preserving ✓ — but applying **one
ordinary `_mutate_params` step at σ = 0.06 makes 9 of 40 exceed parent RMSE + 5°**
(worst 69.7° against a parent's 25.2°). Non-lethal in the generation it is born,
lethal in the next. §4.3's actual complaint — "the mutation is lethal, so the
search can never reach antagonist topologies" — stands. Gate the rate limit on
*transmitted force*, not nameplate capacity, or use a capacity-relative
threshold.

### H6 — `four_bar` is now gated out of existence

Measured: of 120 random `four_bar` genomes, 23 are builder-valid and **22 of
those 23 are geometry-rejected** — roughly **1 in 120 clears the gate**. My
`auto` run showed only **2 species at generation 0** despite 35% four-bar
seeding.

The gate is not wrong; the *seeding* is. Random link-length parameterisation
essentially never produces a feasible four-bar. This needs constrained sampling
at seed time — draw link lengths satisfying Grashof and the envelope, or
rejection-sample until valid — **not** a looser gate. Otherwise R2's correct gate
has quietly removed the only alternative topology the search had.

---

## Medium

- **`mobility()` is still three hardcoded constants** (`return 3 * (2 - 1) - 2 * 1`)
  that never read the genome. `GraphSpec.kutzbach()` is real, but the
  topology-level method the test exercises cannot fail, and E-2's Jacobian-rank
  clause is unimplemented. G10 is closed cosmetically.
- **`cond` is constant for two of three topologies.** Real for `four_bar`
  (56.5–96.7, std 11.3); identically 1.0 for `direct_linear` and `rotary_direct`
  (`interpreter.py:191`, only overwritten in the coupler branch). E-5 says a
  constant is a test failure; `test_conditioning_is_measured_not_constant` picks
  the one topology where it holds.
- **E-8 untouched.** No gear-ratio gene, no reflected inertia — only a docstring
  mention. Excluding rotary from `auto` seeding is the right interim call, but
  the free lunch is still there for anyone passing `--topology rotary_direct`.
- **Immigrant overshoot.** Tail starvation is fixed, but per-species `int()`
  truncation dumps residual slots into immigration: **16 immigrants at 18
  species against an intended 9** — 13% of every generation is a random restart.
  Use largest-remainder. At very high species counts the unconditional
  species-elite append also still overflows and truncates.
- **Hold-out masks diverge for recordings > 12 s.** Evolution builds `ts` with
  `max_duration = 12`; reporting builds `t_full` with 60; `_interleaved_mask`
  slices by fraction of each array's own span. Measured 61% agreement, with
  **37.7% of reported hold-out samples having been in the evolved fit window**.
  Invisible on the 8 s demo.
- **`u_demand_peak` still uses the greedy formula while the winner runs the QP.**
  Measured on an unsaturated 2-actuator design: demand **2.94×** applied peak on
  `a0` with nothing saturating.
- **`N_GEOM_SAMPLES = 16`** against a 128-point table — a crossing in a narrow
  window is missable.
- **Branch threshold mixes units.** `‖ΔX‖` norms metres and radians together,
  and `BRANCH_FLOOR = 0.15` therefore means something different per mechanism
  scale. The spec asked for `BRANCH_K·‖x′‖·Δθ`.
- **First grid point has no warm start** where a topology passes no `X0`
  (`X = zeros`). A wrong-branch start is self-consistent thereafter, so
  `flipped` never fires and the whole sweep is marked valid on the wrong
  assembly.

---

## Test quality

Several new tests assert materially less than the requirement they name:

| test | what it really checks |
|---|---|
| `test_secondary_capacity_decodes_to_zero_rate_g5` | `(0,0) → (0,0)`, automatic for any `g²·const` form. Never tests the `F=0, v=random` configuration `add_antagonist` actually creates — the exact defect. |
| `test_energy_charges_co_contraction_g6` | Still passes with η all 1.0 and hold power all 0.0, so it covers neither efficiency nor hold. Substitutes an allocator the shipped code never produces for §13's stated criterion. |
| `test_demand_equals_applied_when_unsaturated` | Algebraically tautological for `n_act = 1`; doesn't test the saturated case or the QP path where H4's 2.94× mismatch lives. |
| `test_four_bar_interpreter_matches_oracle_to_1e10` | `q`/`r` to 1e-10 ✓, but `I_eff` to 5e-3 and `G` to 5e-2 — 7–9 orders looser than the name and than §13. |
| `test_gate_passes_a_known_good_design` | One hand-picked seed-0 genome; measured gate pass rate is 2.83%. |
| `test_gate_rejects_four_bar_geometry_v8_v9` | `geom_rejected > 0` passes on 1 of 23 *and* on 23 of 23; nothing asserts any four-bar survives — which is why H6 went unnoticed. |
| `test_species_batched_budget` | Threshold 60× looser than the spec, and currently failing. |

Two are genuinely good and I mutation-tested both: reverting the `active`
exclusion in `reduced_sim.py` fails
`test_structural_mutations_behaviour_preserving_20_seeds`, and reverting
`_spring_terms` fails `test_spring_decode_grid_independent_e13`. That is the
standard the others should meet.

---

## Recommended order

1. **H1 (a)–(c)** — early-exit Newton, cheaper `_rot`/`_drot`, drop the
   determinant. This unblocks everything else, including a test suite that
   currently cannot finish. Then amend §2.5's budget to the honest number and
   retune the test to it.
2. **H2** — `pose_points` into `REQUIRED_METHODS`, returning bone segments too.
   One line of enforcement plus two small implementations.
3. **H6** — constrained four-bar seeding, or R2's gate has silently reduced the
   search to one topology.
4. **H4/H5** — charge per-actuator work against a real η and hold power drawn
   from `actuator_design`'s existing screw classes; make the rate exclusion
   capacity-relative.
5. **H3** — the structural operators, and the decision that precedes them:
   moving `graph_spec_batch` construction out of the topology classes and into
   an interpretation of `genome.bodies`/`genome.joints`. Until that happens the
   engine is a curated library with a generic solver, and the README and specs
   should say so.
6. Only then R5–R7.
