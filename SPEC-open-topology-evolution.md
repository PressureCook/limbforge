# Specification — open-topology mechanism evolution

**Status:** draft for review · **Version:** 0.1 · **Date:** 2026-08-12
**Supersedes:** the fixed 6-gene actuator model in `mechanism.py` / `evolve.py`
**Related:** `REVIEW.md` (C1, C2, C3, M5 are blocking prerequisites — see §2)

---

## 1. Purpose and scope

### 1.1 What changes

Today the pipeline searches a **single, hardcoded architecture**: one linear actuator
spanning a revolute joint, anchored at polar `(a, α)` on the proximal segment and
`(b, β)` on the distal segment. `mechanism.decode` maps a 6-gene genome onto that one
topology and nothing else is reachable. The GA tunes a mechanism it was told to build.

This specification replaces that with an **open topology search**. A candidate design
becomes a *planar mechanism graph* — rigid bodies, kinematic joints, actuators,
springs — which the evolutionary loop invents and mutates structurally as well as
parametrically. The search can discover four-bar and six-bar linkages, cam-follower
drives, tendon-and-pulley routes, antagonistic actuator pairs, series elasticity,
geared rotary drives, and combinations of these, without any of them being enumerated
in advance.

The evolved designs must remain **first-class simulation objects**: every topology the
search produces is physically simulated against the recorded trajectory and against
the usage task, using the same controller, saturation and payload model as today, and
can be serialized, reloaded and re-simulated.

### 1.2 What stays the same

- The **anatomical joint is a given**. Every mechanism drives one revolute output
  joint at the origin, between a fixed proximal body and a moving distal body, with the
  flexion convention of `joints.py` unchanged.
- The **pipeline shape** is unchanged: capture → pose → evolve → simulate → actuator
  spec → printables → visualise → report.
- The **plane** is unchanged: 2D sagittal.
- The **controller** is unchanged: PD with gravity feed-forward, force and speed
  saturation.

### 1.3 Compatibility position

Per the decision recorded at kickoff: **every topology admitted to the population must
support the full downstream chain.** A topology that cannot emit an actuator
specification or a printable solid model is not registrable. This is enforced at
registration time (§7.5), not discovered at runtime. There are no dead-end designs and
no "spec not available for this topology" states in the UI.

---

## 2. Blocking prerequisites

These are defects in the current physics. They must land **before** this work, not
alongside it. Rationale: an open search does not tolerate a wrong objective the way a
2-parameter search does. Where a fixed architecture absorbs a modelling error into
slightly-off anchor positions, an open search will restructure the entire mechanism to
exploit it, and the resulting topology will be confidently, unrecoverably wrong.

| ID | Defect | Why it blocks |
|---|---|---|
| **C1** | `simulate.gravity_torque` restores toward θ=0 for every joint; the shoulder's stable pose is θ≈170° and the ankle's is θ≈90° | The generalised gravity term `G(θ)` (§5.4) is built by summing over *all* bodies in the graph. A wrong base convention propagates into every topology, and the search will invent linkages whose whole purpose is to exploit a gravity field pointing the wrong way. |
| **C3** | Shoulder and hip use whole-limb mass at first-segment length; static load understated 1.7× and 2.0× | Topology choice is load-driven. A four-bar that beats a direct linear drive at 24 N·m may lose at 48 N·m, and vice versa. Searching against half the real load selects the wrong architecture, not merely the wrong dimensions. |
| **C2** | `P_mech = F·v` is sign-inverted relative to the codebase's own `τ = −T·dL/dθ` convention | Energy becomes a **fitness term** in this work (§8.2) — it is the primary discriminator between multi-actuator and single-actuator topologies. Evolving against an inverted power sign would systematically prefer the wrong ones. |
| **M5** | `F_peak` is measured from an already-clipped signal, so it can never exceed `F_max` | Fitness must rank *how badly* a design saturates in order to gradient toward topologies with better transmission. A metric that saturates at 100% gives the search no signal to follow. `simulate_population` must record unclipped demand alongside the clipped applied force. |

**Prerequisite acceptance:** a test that releases each joint from rest under gravity
alone and asserts the settling angle matches the anatomical resting pose
(elbow/knee/hip 0°, shoulder 180°, ankle 90°); a test asserting
`sign(P_mech) == sign(τ_act · ω)` on a known trajectory; a test asserting
`F_demand_peak ≥ F_applied_peak` with strict inequality on a deliberately
under-sized design.

---

## 3. Terminology

| Term | Meaning |
|---|---|
| **Output joint** | The anatomical revolute joint at the origin. Its angle θ is the flexion coordinate of `joints.py`. |
| **Ground body** | The proximal segment (socket side). Fixed frame. |
| **Output body** | The distal segment. Its orientation is rigidly tied to θ. |
| **Coupler** | Any additional rigid body introduced by the topology. |
| **Mechanism graph** | Bodies (nodes) + kinematic joints, actuators and springs (edges). |
| **Mobility** | Kinematic degrees of freedom of the linkage, excluding actuator inputs. |
| **Transmission table** | Precomputed θ-indexed map of every quantity the simulator needs (§5.3). This is the *only* interface between topology and dynamics. |
| **Transmission ratio** `r_i(θ)` | Generalised moment arm of actuator *i*: `r_i = −dq_i/dθ`, where `q_i` is the actuator's own coordinate (length for linear/tendon, angle for rotary). Exactly generalises today's `−dL/dθ`. |
| **Species** | A cluster of genomes sharing a topology, in the NEAT sense (§6.4). |
| **Branch** | One of the discrete assembly configurations a closed-loop linkage admits for a given input angle. |

---

## 4. Requirements

Numbered, testable. **MUST** = acceptance-blocking.

### Modelling

- **R-1** The system MUST represent a design as a planar mechanism graph of rigid
  bodies, kinematic joints, actuators and passive elements, serialisable to and from
  JSON without loss.
- **R-2** The system MUST support at least these actuator kinds: `linear`
  (screw/EMA, bidirectional force along a line), `rotary` (geared motor applying
  torque about a revolute joint), `tendon` (unilateral, tension-only, routed over
  via-points). Each carries its own capacity genes.
- **R-3** The system MUST support passive `spring` (linear or torsional, stiffness and
  free length/angle) and `hard_stop` elements.
- **R-4** The system MUST admit redundantly actuated topologies (e.g. antagonistic
  pairs) and resolve the force distribution deterministically (§5.6).
- **R-5** Mechanism mobility MUST be verified numerically from the rank of the
  constraint Jacobian, not only from the Kutzbach count. (Kutzbach is necessary, not
  sufficient — overconstrained and paradoxical linkages pass it and do not move.)

### Simulation

- **R-6** Every registrable topology MUST reduce to a single generalised coordinate θ
  and MUST expose its dynamics solely through a transmission table (§5.3).
- **R-7** The simulator MUST integrate the exact reduced equation of motion of §5.4,
  including the `½ (dI_eff/dθ) θ̇²` term. (Today's model legitimately omits it because
  `I_eff` is constant; with couplers it is not.)
- **R-8** A design loaded from `design.json` MUST re-simulate to traces identical to
  the originating run within 1e-9 relative tolerance, on the same seed and inputs.
- **R-9** The simulator MUST reject, before integration, any candidate whose
  transmission table is invalid anywhere in the required ROM (§5.5). Rejection MUST be
  a hard gate, not a soft penalty.
- **R-10** `simulate_population` MUST remain vectorised over the population axis. Per
  candidate Python loops in the inner time loop are not acceptable (§10).

### Search

- **R-11** The genome MUST be variable-length and typed, with structural genes carrying
  global innovation identifiers for alignment (§6.1).
- **R-12** Structural mutations MUST be behaviour-preserving at birth (§6.3): a newly
  added element MUST be initialised so the mechanism's transmission table is unchanged
  to within a stated tolerance.
- **R-13** The population MUST be speciated, with fitness sharing, so a newly invented
  topology is protected long enough to be parameter-optimised before competing
  globally (§6.4).
- **R-14** Crossover MUST only occur between genomes in the same species, aligned by
  innovation id.
- **R-15** The initial population MUST include the legacy direct-linear topology, and
  the engine MUST reproduce legacy-quality results when restricted to it (§12, Phase 1
  acceptance).
- **R-16** Fitness MUST include an explicit complexity cost over bodies, joints and
  actuators (§8.2). Without it the search bloats.

### Downstream

- **R-17** Every registered topology MUST implement the full capability interface of
  §7.1. Registration MUST fail loudly at import time if any method is missing.
- **R-18** `actuator_design` MUST accept a list of `ActuatorRequirement` objects rather
  than a `Mechanism`, and MUST produce one spec section per actuator (§7.2).
- **R-19** `actuator_design` MUST gain a rotary sizing path (gearbox ratio, output
  torque, backlash class, mass) alongside the existing screw path.
- **R-20** `printable` MUST build parts from a declarative primitive list (§7.3), so
  any topology composed of registered primitives gets STLs without bespoke code.
- **R-21** `design.json` MUST be versioned; the UI and `runs/index.json` MUST render
  both legacy and graph designs.

---

## 5. Architecture

### 5.1 The mechanism graph

```
Body        id, kind ∈ {ground, output, coupler}, mass, com (local), inertia,
            length (couplers only — mass follows from length × linear density)

Joint       id, kind ∈ {revolute, prismatic}, bodyA, anchorA (local xy),
            bodyB, anchorB (local xy), innovation

Actuator    id, kind ∈ {linear, rotary, tendon},
            mount:  linear/tendon → (bodyA, ptA, bodyB, ptB[, via_points])
                    rotary        → (joint_id)
            capacity genes: linear/tendon → F_max, v_max
                            rotary        → tau_max, omega_max
            innovation

Spring      id, kind ∈ {linear, torsional}, same mount forms, k, q_free, innovation

HardStop    joint_id, q_min, q_max
```

Two bodies are structural invariants and cannot be removed: `GROUND` (the proximal
segment) and `OUTPUT` (the distal segment). One joint is invariant: the output
revolute at the origin between them. Everything else is under evolutionary control.

The legacy design is exactly the graph:

```
bodies:    [GROUND, OUTPUT]
joints:    [revolute(GROUND, (0,0), OUTPUT, (0,0))]          # the anatomical joint
actuators: [linear(GROUND, A(a,α), OUTPUT, B(b,β), F_max, v_max)]
```

which is what R-15 requires the seeded population to contain.

### 5.2 The single-DOF reduction — the central design decision

A general mechanism graph is a constrained multibody system, normally simulated as an
index-3 DAE with Baumgarte stabilisation. That is the wrong tool here: it cannot be
vectorised over a population of 160 structurally different candidates across 2,880
timesteps, and the GA would become intractable.

Instead, **constrain the search to mechanisms of mobility 1** (R-5). For a single-DOF
mechanism, every body's position and orientation is a function of θ *alone*. That
means all kinematics can be precomputed once per candidate on a θ grid, and the
dynamics collapse to a scalar ODE in θ — precisely the shape `simulate_population`
already integrates.

This buys three things at once:

1. **The existing simulator survives.** The PD controller, gravity feed-forward,
   saturation logic, payload model and tracking metric are unchanged in structure.
   Only the coefficients become θ-dependent.
2. **Vectorisation survives.** After precomputation each candidate is a set of
   `(N,)` lookup tables; a population is `(P, N)`. The time loop becomes table gathers
   and elementwise arithmetic — fully vectorised (R-10).
3. **Moment arms generalise exactly.** `r_i(θ) = −dq_i/dθ` reduces to today's
   `−dL/dθ` for the legacy topology, so the virtual-work identity that
   `selftest.test_geometry` already verifies extends unchanged and can be tested the
   same way.

Mobility > 1 mechanisms (genuinely multi-DOF linkages) are **out of scope** (§14).
Redundant *actuation* of a mobility-1 mechanism is in scope and is a different thing
(§5.6).

### 5.3 The transmission table

The sole interface between topology and dynamics.

```
TransmissionTable
  theta_grid   (N,)        uniform grid over [θ_lo − pad, θ_hi + pad]
  q            (n_act, N)  each actuator's own coordinate
  r            (n_act, N)  transmission ratios, r_i = −dq_i/dθ
  I_eff        (N,)        reduced inertia about the output joint
  dI_dtheta    (N,)        analytic where available, else spectral difference
  G            (N,)        generalised gravity torque from all moving bodies
  S            (N,)        generalised spring torque from all passive elements
  valid        (N,) bool   see §5.5
  cond         (N,)        constraint-Jacobian condition number (diagnostic)
```

Construction, per candidate:

1. Build the loop-closure residual `f(x, θ) = 0`, where `x` holds the independent
   coordinates of the coupler bodies. For a mobility-1 linkage `dim(x) = 3·n_coupler −
   (constraints)`, typically 1–3.
2. Sweep the grid, solving with **Newton, warm-started from the previous grid point**
   and an **analytic Jacobian**. Warm starting is what makes this cheap: 2–4 iterations
   per point after the first.
3. Differentiate: `r_i = −dq_i/dθ` and `dI_eff/dθ` from the same solve. Prefer the
   implicit-function-theorem derivative (`dx/dθ = −J_x⁻¹ J_θ`, reusing the factorised
   Jacobian) over finite differences — it is both cheaper and cleaner near
   ill-conditioned points.
4. Accumulate `I_eff`, `G`, `S` by summing over bodies and passive elements (§5.4).

**Measured cost.** A prototype four-bar solve — 128 grid points, one unknown, analytic
Jacobian, warm-started — runs at **3.5 ms per candidate** as a scalar Python loop.
Unbatched that is 160 × 120 × 3.5 ms ≈ **77 s per joint**, which is not acceptable.
Batching the Newton iteration over the *species* (all candidates in a species share the
same residual equations, differing only in parameters) turns it into ~4 vectorised
Newton sweeps over `(P, N)` arrays and brings it into the low milliseconds per
generation. **Species-batched precomputation is therefore a requirement, not an
optimisation** (§10).

### 5.4 Reduced equation of motion

Generalised coordinate θ. Kinetic energy `T = ½ I_eff(θ) θ̇²` with

```
I_eff(θ) = Σ_k [ m_k · |∂p_k/∂θ|²  +  I_k · (∂φ_k/∂θ)² ]
```

summed over all moving bodies *k* (output segment, couplers, and the payload point
mass), where `p_k` is the body COM and `φ_k` its orientation. Lagrange's equation for a
single-DOF system gives, exactly:

```
I_eff(θ)·θ̈  +  ½ (dI_eff/dθ)·θ̇²  =  G(θ) + S(θ) + Σ_i r_i(θ)·u_i  −  b·θ̇
```

with

```
G(θ) = −Σ_k m_k · g · (dy_k/dθ)                     generalised gravity
S(θ) = Σ_j k_j · (q_j(θ) − q_free,j) · r_j(θ)       generalised spring torque
u_i                                                  actuator effort (tension-positive
                                                     for linear/tendon, torque for rotary)
```

Notes:

- The `½ (dI_eff/dθ) θ̇²` term is **mandatory** (R-7). The current code omits it
  legitimately — with no couplers, `I_eff = m·Ld²/3 + payload·Ld²` is constant in θ, so
  the term is identically zero. Once a coupler exists it is not, and omitting it is a
  silent modelling error that the search will exploit.
- Setting `n_coupler = 0`, one linear actuator, no springs recovers the current
  equation term for term. This is the Phase 1 regression check (§12).
- `G(θ)` replaces `simulate.gravity_torque` entirely. The C1 fix (per-joint neutral
  offset) belongs in how the *output body*'s `dy/dθ` is computed, and then flows to all
  bodies for free — which is the argument for landing C1 first (§2).
- Sign convention: `r_i ≡ −dq_i/dθ` and `τ_i = r_i·u_i`, matching the existing
  `tau_act = -T_ax * dL`.

### 5.5 Validity gate

A grid point is **invalid** if any of:

| Condition | Threshold | Why |
|---|---|---|
| Newton failed to converge | ‖f‖ > 1e-9 after 20 iters | linkage does not assemble here |
| Constraint Jacobian singular | `cond(J) > 1e8` | dead centre / change point |
| Branch flip | `‖x_k − x_{k−1}‖ > branch_tol` | solution jumped to another assembly |
| Transmission angle out of range | μ ∉ [30°, 150°] | linkage jams or force-multiplies pathologically |
| Link self-intersection | any two non-adjacent links cross | mechanism cannot be built |
| Body outside limb envelope | outside capsule around the segments | cannot be packaged on a limb |

If any grid point inside `[θ_lo, θ_hi]` is invalid, the candidate is **rejected before
simulation** (R-9) and assigned a large constant cost plus a small gradient term
(fraction of ROM covered) so the search can still climb toward feasibility.

This gate is not optional bookkeeping. In the prototype, a four-bar with plausible-
looking link lengths passed through a dead centre inside the ROM and produced
transmission ratios swinging from **−151 mm to +335 mm with sign reversals**. Simulated
naively, such a design scores *well* — infinite mechanical advantage at the singularity
flatters the tracking error — while being physically useless. Hard rejection before
integration is what stops the search colonising these.

### 5.6 Force allocation under redundant actuation

A mobility-1 mechanism with `n_act > 1` (antagonistic pairs, hybrid linear+spring
assist) is statically indeterminate: the commanded torque `τ_cmd` can be produced by
many effort vectors. The allocation policy must be deterministic (R-4).

**Exact policy** — a small box-constrained QP per timestep:

```
minimise   Σ_i w_i u_i²
subject to Σ_i r_i(θ) u_i = τ_cmd
           |u_i| ≤ u_max,i        (0 ≤ u_i ≤ u_max,i for tendons)
```

**Population policy** — the QP is far too slow inside a vectorised population loop.
During evolution, use *capacity-weighted greedy allocation*: distribute `τ_cmd`
proportional to each actuator's torque capacity `|r_i|·u_max,i`, clip to limits, then
redistribute the residual once among unsaturated actuators. O(n_act), fully
vectorisable, and exact for the single-actuator case.

**This discrepancy must be measured, not assumed away.** `simulate_final` uses the
exact QP. A validation test MUST assert that, for the winning design of each run, the
greedy and QP allocations agree within 0.5° tracking RMSE and 5% peak effort; if they
diverge beyond that, the run reports it and the design is re-ranked under the QP. This
is the honest version of the tradeoff and it belongs in the acceptance criteria (§12).

---

## 6. Genome and evolutionary operators

### 6.1 Representation

Fixed-length real vectors cannot encode variable topology, and naive variable-length
crossover between different topologies produces non-viable offspring — the classic
failure mode of structural GA. The design follows **NEAT**: structural genes carry
global **innovation identifiers** issued by a monotonic counter, so two genomes can be
aligned by shared history rather than by position.

```
Genome
  bodies    : list[BodyGene]        # coupler bodies only; GROUND/OUTPUT implicit
  joints    : list[JointGene]       # each with .innovation
  actuators : list[ActuatorGene]    # each with .innovation
  springs   : list[SpringGene]      # each with .innovation
  meta      : {seed_topology, parent_ids, birth_generation}
```

Continuous genes remain in `[0,1]` and decode through the existing bounded/log-scale
scheme of `mechanism.decode`, extended per gene type. Keeping the `[0,1]` convention
means the existing BLX-α crossover and decaying-σ Gaussian mutation carry over
unchanged for the parametric part.

### 6.2 Reachable design space

Topologies the operator set can construct, none of which are enumerated anywhere in
code:

- direct linear actuator (the legacy design)
- four-bar and six-bar driven linkages, including slider-crank
- cam-follower drives (cam profile as a coupler body with a polar-radius gene vector)
- tendon-over-pulley routes with one or more via-points
- antagonistic actuator pairs (agonist/antagonist tendons or linear pairs)
- rotary geared drive at the output joint, with or without a reduction linkage
- series elastic actuation (spring in series with a linear actuator)
- parallel spring assist / gravity compensation
- any composition of the above

### 6.3 Structural mutation operators

Every structural mutation MUST be **behaviour-preserving at birth** (R-12). Without
this, each structural change is effectively lethal and topology search never gets off
the ground — the new variant dies before its parameters can be tuned.

| Operator | Action | Behaviour-preserving initialisation |
|---|---|---|
| `add_spring` | attach a spring between two existing points | `k = 0` |
| `add_actuator` | attach a second actuator | `u_max = 0` |
| `split_link` | replace a binary link with two links + a coupler + pin | new pin placed **collinear** on the original line; transmission table unchanged to within tolerance |
| `add_via_point` | route a tendon over a new pulley | pulley placed **on** the existing tendon line, radius 0⁺ |
| `retype_actuator` | linear ↔ rotary ↔ tendon | capacity matched so peak output torque is preserved at mid-ROM |
| `add_coupler_loop` | close a new kinematic loop | admitted only if the resulting mobility is still 1 and the loop assembles across the whole ROM |
| `remove_*` | delete an element | not behaviour-preserving; gated on the element's measured contribution being below a threshold |
| `simplify` | merge collinear links, drop zero-stiffness springs and zero-capacity actuators | strictly behaviour-preserving; run periodically to control bloat |

A unit test MUST assert, for each preserving operator, that the pre- and post-mutation
transmission tables agree to within 1e-6 across the ROM.

### 6.4 Speciation

Genomes are clustered by a **topological distance**:

```
δ(g1, g2) = c1·(excess innovations)/N + c2·(disjoint innovations)/N
          + c3·(mean |Δ| of matched continuous genes)
```

with the usual NEAT semantics. Genomes within `δ_threshold` form a species; explicit
fitness sharing divides each individual's fitness by its species size; species are
allocated offspring proportional to mean adjusted fitness; species with no improvement
for `stagnation_limit` generations are retired.

Two reasons this is required rather than nice-to-have (R-13):

1. **Protection.** A freshly invented four-bar starts with untuned link lengths and
   will lose to a well-tuned direct-linear design every time. Without sheltered
   competition, the search converges to the seeded topology and this entire project
   delivers nothing.
2. **Batching.** Species share residual equations, which is exactly what makes the
   vectorised Newton precompute of §5.3 possible. Speciation is simultaneously the
   diversity mechanism and the performance mechanism.

### 6.5 Seeding and immigration

- Generation 0 contains the legacy direct-linear topology across a spread of
  parameters (R-15), guaranteeing the new engine starts no worse than the old one.
- A configurable fraction of each generation is **immigrant**: freshly generated
  random topologies from a small archetype library (four-bar, slider-crank,
  tendon+pulley, rotary geared, antagonistic pair), parameterised randomly and
  filtered through the validity gate. This keeps structural diversity from collapsing
  in long runs.
- The archetype library is a *seeding convenience only*. It is not the design space,
  and topologies outside it must remain reachable through mutation.

---

## 7. Capability interface and downstream changes

### 7.1 The `Topology` protocol

Registration-time enforced (R-17). Import of a topology module that fails to implement
every method raises immediately.

```python
class Topology(Protocol):
    kind: str                       # stable identifier, used in design.json

    # --- kinematics / dynamics -------------------------------------------
    def build_table(self, genome, ctx, theta_grid) -> TransmissionTable: ...
    def mobility(self, genome) -> int: ...

    # --- downstream capabilities (all mandatory) --------------------------
    def actuator_requirements(self, genome, traces) -> list[ActuatorRequirement]: ...
    def solid_model(self, genome, scale) -> list[Part]: ...
    def draw(self, ax, genome, theta) -> None: ...
    def describe(self, genome) -> dict: ...
```

Batched variants (`build_table_batch` over a species) are optional but strongly
recommended; the engine falls back to a Python loop if absent and logs the cost.

### 7.2 `actuator_design` changes

Currently `design_actuator(mech, final, ctx)` reads `mech.F_max, v_max, stroke, L_ret`
directly. This must be refactored to consume a **per-actuator requirement**:

```python
@dataclass
class ActuatorRequirement:
    kind: str                  # "linear" | "rotary" | "tendon"
    peak_effort: float         # N (linear/tendon) or N·m (rotary)
    rms_effort: float
    peak_rate: float           # m/s or rad/s
    travel: float              # stroke (m) or sweep (rad)
    effort_trace: np.ndarray   # for the energy integral
    rate_trace: np.ndarray
    packaging_envelope: float  # available retracted length or diameter
```

Changes required:

- **Split the sizing paths.** The existing screw/gear/motor chain serves `linear` and
  `tendon`. A **new rotary path** (R-19) is genuine new work: gearbox ratio selection,
  output torque and backlash class, planetary vs harmonic stage efficiency, mass
  estimate. Budget this as its own work item — it is not a refactor.
- **One spec section per actuator.** `spec_markdown` and `draw_actuator` currently
  assume exactly one. Both need a per-actuator loop, and the recommendation logic
  (leadscrew vs ballscrew) becomes per-actuator.
- **System-level rollup.** Total mass, total energy per movement, and combined battery
  estimate must be summed across actuators — for antagonistic topologies this is the
  number that decides whether the design is viable at all.
- **Fix C2 first** (§2). The energy integral is about to be summed over multiple
  actuators and used as a fitness term; the sign error must not be multiplied.

### 7.3 `printable` changes

Replace bespoke geometry with a declarative primitive list (R-20):

```python
Part = Plate(outline, bores)      # linkage plates, cam plates, brackets
     | Pin(d, length, shoulder)
     | Barrel(od, bore, length, motor_pod)
     | Rod(d, length)
     | Pulley(d, groove_profile)
     | CamPlate(profile_polar)
```

`solid_model` returns a list of these; the generator turns them into watertight meshes,
lays out the print plate, and renders the preview — all topology-agnostic. A topology
introducing a genuinely new primitive adds it to this list once, and every future
topology can use it.

Also fix the latent defect noted in the review: `printable` currently builds the
actuator mock at `body_len` while the link plates use true anchor positions. These
coincide today only because the GA parks exactly on the packaging constraint. Under
open topology that coincidence disappears and the printed parts will not assemble.
Build the mock at the actual retracted length.

### 7.4 `visualize` and `runner` changes

- `plot_mechanism_curves` becomes topology-driven: one transmission-ratio trace per
  actuator, plus `I_eff(θ)` and the `G+S` load curve, drawn from the transmission
  table rather than from `length_and_derivative`.
- `make_gif` / `make_limb_gif` delegate frame drawing to `Topology.draw`, which
  receives an axis and θ and renders whatever the mechanism actually is.
- `_mech_anchors_world` (currently hardcoded to one A/B pair, with a per-joint
  `MECH_FLIP` table) is replaced by a general body-pose query on the transmission
  table.
- `runner._design_dict` emits the graph document of §9 instead of `m.to_dict()`.

### 7.5 Registration

```python
@register_topology("four_bar")
class FourBar:  ...
```

The decorator MUST verify every protocol method exists and MUST run a smoke test
(build a random genome, produce a table, produce requirements, produce a solid model)
at import time. A topology that cannot complete the chain never enters the population
— this is the mechanism by which the compatibility position of §1.3 is enforced.

---

## 8. Fitness and constraints

### 8.1 What carries over

The existing cost structure is sound and is retained:

```
cost = W_TRACK·rmse_deg + W_FORCE·(effort class) + W_SPEED·(rate class)
     + W_SIZE·(compactness) + penalties
```

### 8.2 New terms

| Term | Form | Why |
|---|---|---|
| **Complexity** | `W_PARTS·n_bodies + W_JOINTS·n_joints + W_ACT·n_actuators` | R-16. Open structural search bloats without an explicit parts cost. This is the single most important new term. |
| **Energy** | `W_ENERGY · E_per_movement` from the (C2-corrected) integral | The primary discriminator between multi-actuator and single-actuator topologies. A spring-assisted design wins here or it does not win at all. |
| **System mass** | `W_MASS · Σ actuator + structure mass` | Otherwise the search adds actuators freely to buy tracking error. |
| **Transmission-angle margin** | penalty on `min over ROM of angular distance of μ from {0°,180°}` | Classic linkage quality metric; keeps designs off the edge of jamming. |
| **Singularity margin** | penalty on `min over ROM of 1/cond(J)` | Distance from the dead centres that §5.5 rejects outright. Gives gradient *toward* feasibility rather than a cliff. |
| **Unclipped saturation** | `Σ relu(|u_demand| − u_max)` | Requires the M5 fix. This is what lets the search tell a slightly-undersized design from a hopeless one. |
| **Packaging** | every body inside the limb envelope across the whole ROM | Generalises today's retracted-length check to arbitrary geometry. |

### 8.3 Tolerance robustness — second-stage screen

Open topology search has a strong preference for razor-edge geometries that work only
at exact nominal dimensions. Evaluating robustness per individual is too expensive, so
it runs as a **screen on the top-K designs of each generation**:

- perturb link lengths by ±0.3 mm and pin positions by ±0.2 mm (printed-part
  tolerances), N=16 samples
- re-evaluate tracking RMSE and validity
- penalise the *variance*, and hard-reject any design that becomes invalid under
  perturbation

A design that only assembles at nominal dimensions is not a design. This screen is what
separates "the simulator liked it" from "you could build it".

### 8.4 Overfitting control

More design freedom means more capacity to overfit the recorded trajectory. The
existing usage-task re-evolution loop (`runner._process_joint`) partially addresses
this, but under open topology it must become a **hold-out**:

- split the recorded trajectory into fit and hold-out segments (e.g. alternate
  repetitions)
- evolve on the fit segment
- report tracking error on the hold-out segment in `report.md` and `design.json`
- a large fit/hold-out gap is a warning surfaced in the UI

---

## 9. Data formats

### 9.1 `design.json` v2

```jsonc
{
  "schema_version": 2,
  "topology": {
    "kind": "four_bar_linear",
    "bodies":    [ { "id": "gnd", "kind": "ground" }, … ],
    "joints":    [ { "id": "j0", "kind": "revolute", "a": ["gnd", [0,0]], … } ],
    "actuators": [ { "id": "a0", "kind": "linear", "mount": …, "F_max": 412.0, … } ],
    "springs":   [ … ]
  },
  "transmission_summary": {
    "r_range_mm": { "a0": [11.4, 27.8] },
    "I_eff_range_kg_m2": [0.041, 0.058],
    "min_transmission_angle_deg": 58.2,
    "min_reciprocal_condition": 0.0071
  },
  "performance": { "rmse_deg": …, "rmse_holdout_deg": …, "effort_peak": …,
                   "effort_demand_peak": …, "energy_J": … },
  "evolution": { "species_id": …, "birth_generation": …, "innovations": [ … ] },
  "robustness": { "rmse_std_deg": 0.14, "invalid_fraction": 0.0 }
}
```

- `schema_version: 1` is the current format. Readers MUST handle both (R-21); a v1
  document is interpreted as the legacy direct-linear topology.
- The topology block MUST be sufficient to reconstruct and re-simulate the design
  (R-8), with no reference to run-local state.

### 9.2 UI

`runs/index.json` gains `topology_kind` and `n_actuators`. `run.html` gains a topology
chip and, per the review's finding M7, must surface `packaging: DOES NOT FIT` and the
re-evolution flag on single-joint runs as well as limb runs.

---

## 10. Performance budget

Current cost, single joint: population 160 × 120 generations, each evaluation
integrating ~2,880 timesteps vectorised over the population.

| Stage | Budget | Notes |
|---|---|---|
| Table precompute, species-batched | ≤ 15% of generation time | Measured scalar cost 3.5 ms/candidate ⇒ 77 s/joint unbatched. **Batching is mandatory**, not an optimisation. |
| Elite table caching | — | Elites are structurally unchanged; cache by `(genome_hash)`. |
| Validity gate | ≤ 2% | Runs before integration and prunes most random topologies cheaply. |
| Integration | ~unchanged | Same loop shape, table gathers instead of closed-form `length_and_derivative`. |
| Robustness screen | ≤ 10% | Top-K only, every generation or every G generations. |
| **Total target** | **≤ 1.35× current wall clock** | For equal population and generations. |

Table gathers must use a manual linear-interpolation gather over `(P, N)` arrays
(`np.interp` is 1-D only and would force a Python loop, violating R-10).

If the budget cannot be met, the fallback is to reduce the θ grid from 128 to 64
points and re-validate the transmission-ratio error against a 512-point reference,
rather than to weaken the validity gate.

---

## 11. Validation plan

### Unit

- **Reduction correctness.** For the legacy topology, the transmission table's `r(θ)`
  MUST equal `−dL/dθ` from `mechanism.length_and_derivative` to 1e-10. This extends the
  existing `selftest.test_geometry` finite-difference check rather than replacing it.
- **Reduced EOM.** For a two-body pendulum with a known analytic solution, the
  integrator MUST match to 1e-6 over 5 s, *including* the `½ dI/dθ θ̇²` term. A test
  MUST fail if that term is removed.
- **Behaviour preservation.** Each preserving mutation operator (§6.3): pre/post
  transmission tables agree to 1e-6.
- **Validity gate.** A deliberately dead-centred four-bar MUST be rejected. Regression
  fixture: the prototype geometry that produced −151…+335 mm transmission ratios.
- **Force allocation.** Greedy vs QP agreement on an antagonistic pair, within the
  §5.6 tolerance.
- **Round-trip.** R-8: serialise → reload → re-simulate → identical traces.

### Integration

- Full pipeline on the synthetic demo for every archetype, asserting each produces a
  spec, STLs and a GIF.
- Limb run on the synthetic arm and leg demos.
- **Legacy regression:** graph engine restricted to the direct-linear topology, same
  seed, must match the legacy engine's evolved design within tolerance.

### Physical plausibility review

Automated tests cannot tell you whether an evolved cam-follower is a good idea. Every
phase gate includes a **manual review of the ten best designs** by someone who would be
willing to build one. Findings feed back into the §8.2 penalty weights. Budget this as
real time, not a formality — it is the only check that catches "physically valid,
practically absurd".

---

## 12. Phasing and acceptance

### Phase 0 — Prerequisites

Land C1, C2, C3, M5 with the tests of §2.
**Accept when:** the release-from-rest test passes for all five joints and the
shoulder's evolved force class changes as predicted (~140 N → ~300 N on the arm demo).

### Phase 1 — Reduction infrastructure, one topology

Introduce `TransmissionTable`, the reduced integrator, and the `Topology` protocol.
Port the legacy design to it as the first registered topology. No search changes.
**Accept when:** the legacy topology through the new engine reproduces current results
on the demo within 1e-6 RMSE; §11 unit tests pass; wall clock within 1.1× of current.
*This phase ships no new capability and is the most important one — it is where the
correctness of everything downstream is established.*

### Phase 2 — Multi-body kinematics

Loop closure, mobility check, validity gate, species-batched precompute. Add four-bar
and slider-crank as registered topologies with hand-written parameterisations. Still no
structural search — topologies are selected by flag.
**Accept when:** a four-bar can be simulated, specced, printed and animated end to end;
the dead-centre fixture is rejected; performance budget met.

### Phase 3 — Structural search

Variable-length genome, innovation tracking, speciation, mutation operators, complexity
cost.
**Accept when:** on the synthetic demo, the search discovers at least one non-seeded
topology that beats the best direct-linear design on total cost; species diversity
does not collapse before generation 60; bloat is bounded (mean parts count stable over
the last 30 generations).

### Phase 4 — Remaining actuator kinds and capabilities

Rotary sizing path, tendon routing, springs, antagonistic allocation with the QP,
declarative printables, robustness screen, hold-out validation.
**Accept when:** every actuator kind produces a complete spec and printable set; the
robustness screen rejects a known razor-edge design; the greedy/QP discrepancy is
within tolerance on all demo runs.

### Phase 5 — UI, formats, documentation

`design.json` v2, UI rendering, README rewrite, migration of existing runs.
**Accept when:** v1 and v2 runs both render; a v1 run re-simulates through the new
engine.

---

## 13. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Search collapses to the seeded topology** | High — delivers nothing | Speciation with fitness sharing (§6.4); behaviour-preserving mutation (§6.3); immigration (§6.5). Phase 3 acceptance explicitly tests for this. |
| **Bloat** — parts count grows without bound | High | Explicit complexity cost (R-16); periodic `simplify`; Phase 3 acceptance tracks mean parts count. |
| **Precompute cost makes the GA intractable** | High | Species batching (mandatory, §5.3); elite caching; documented fallback to a coarser grid (§10). |
| **Designs that simulate but cannot be built** | High | Validity gate (§5.5); packaging in the envelope; tolerance robustness screen (§8.3); manual review at every phase gate (§11). |
| **Overfitting to the recording** | Medium | Hold-out split (§8.4); usage-task re-evolution retained. |
| **Loop-closure non-convergence on exotic topologies** | Medium | Warm-started Newton with analytic Jacobians; convergence failure is a validity rejection, never an exception. |
| **Rotary sizing path underestimated** | Medium | Called out as new work, not a refactor (§7.2). Do not fold into the Phase 1 estimate. |
| **Non-determinism from variable-length genomes** | Medium | Innovation counter seeded per run; all dict iteration ordered; `run.json` records the full genome. R-8 round-trip test guards it. |
| **Reviewer cannot judge exotic designs** | Medium | Every design ships with the transmission-ratio plot, `I_eff(θ)`, transmission-angle margin and the robustness spread — the numbers a mechanism engineer would ask for. |

---

## 14. Out of scope

Explicitly not in this specification:

- **3D mechanisms.** Planar sagittal only, as today.
- **Mobility > 1 mechanisms.** The single-DOF reduction (§5.2) is load-bearing for
  both correctness and performance. Multi-DOF requires a genuine DAE simulator and is a
  separate project.
- **Multi-DOF anatomical joints.** The output joint stays revolute.
- **Inter-joint coupling.** Joints remain designed independently; reaction forces
  between them are still unmodelled (as the README already states).
- **Control co-design.** The PD-with-gravity-feed-forward controller is fixed. Evolving
  the controller alongside the mechanism is a natural follow-on and would change the
  fitness landscape substantially.
- **Compliant / continuum mechanisms.** Rigid bodies with discrete springs only.
- **Vendor catalogue matching.** Actuator specs remain first-principles estimates.
- **Real-time or hardware-in-the-loop simulation.**

---

## 15. Open questions

1. **θ grid resolution.** 128 points is a starting assumption. Needs an error study
   against a 512-point reference across the archetype library before it is fixed.
2. **Cam representation.** Polar radius vector as genes is the obvious encoding, but
   the gene count is large and mutation is poorly conditioned. A spline or Fourier
   parameterisation may be better — decide during Phase 4.
3. **Should springs count toward the complexity cost at the same weight as
   actuators?** A spring is far cheaper to build than an actuator, so probably not, but
   the ratio needs calibration against the manual review.
4. **Robustness screen cadence.** Every generation on top-K, or every G generations on
   a larger K? Affects the performance budget materially.
5. **Does the hold-out split (§8.4) interact badly with the usage-task re-evolution
   loop?** Both are anti-overfitting mechanisms; they may be redundant or may conflict.
6. **Migration of existing runs.** Re-simulate v1 designs through the new engine and
   overwrite, or leave them frozen and label them legacy? Affects whether historical
   comparisons are meaningful.

---

## Appendix A — Reduction to the current model

Setting `n_coupler = 0`, one linear actuator, no springs, no hard stops:

| Spec quantity | Current code |
|---|---|
| `q_0(θ)` | `L` from `mechanism.length_and_derivative` |
| `r_0(θ) = −dq_0/dθ` | `−dL` (with `tau_act = -T_ax * dL`) |
| `I_eff(θ)` | `ctx.inertia + payload·Ld²` (constant) |
| `dI_eff/dθ` | identically 0 — which is why the term is legitimately absent today |
| `G(θ)` | `simulate.gravity_torque(θ, ctx)` (after the C1 fix) |
| `S(θ)` | 0 |
| Allocation | trivial, `u_0 = τ_cmd / r_0` |
| Validity | `L > 0` always; no loop to close |

Every term maps. This is the Phase 1 regression contract, and the reason Phase 1 ships
no new capability but gates everything that follows.
