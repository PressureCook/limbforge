"""Structural evolution over registered topologies (SPEC §6, §8, Phases 2-4).

- population of typed variable-length genomes, speciated by structural
  signature with fitness sharing and stagnation retirement (§6.4)
- structural mutations behaviour-preserving at birth (§6.3): springs decode to
  k=0 and secondary actuators to zero capacity at gene 0
- species-batched table precompute (§5.3), hard validity gate before
  integration (§5.5) with a feasibility gradient (ROM coverage)
- fitness = tracking + effort/rate class + complexity + energy + mass +
  singularity margin + unclipped saturation (§8.2)
- hold-out split (§8.4): evolve on the first HOLDOUT_FIT fraction of the
  recording, report RMSE on the remainder
- robustness screen (§8.3) on the top designs; the winner must survive it
- the final winner is re-simulated with the exact QP allocation and the
  greedy/QP discrepancy is reported (§5.6)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import topologies  # noqa: F401 - triggers topology registration
from . import opengraph   # noqa: F401 - registers the open "mechgraph" topology
from .gate import apply_gate
from .reduced_sim import TableBatch, simulate_tables
from .simulate import make_target
from .topology import (ActuatorGene, Genome, JointGene, SpringGene, TOPOLOGIES,
                       get_topology, next_innovation)

# fitness weights (§8.1 carried over + §8.2 new terms)
W_TRACK = 2.0
W_FORCE = 0.6      # per 500 N linear / per 25 N m rotary
W_SPEED = 0.4      # per 0.2 m/s linear / per 8 rad/s rotary
W_COMPLEX_BODY = 0.35
W_COMPLEX_JOINT = 0.25
W_COMPLEX_ACT = 1.2
W_COMPLEX_SPRING = 0.10   # springs are far cheaper than actuators (spec Q3)
W_ENERGY = 0.09    # per J over the fit window - primary topology discriminator (§8.2)
W_MASS = 0.8       # per kg
W_SAT = 2.0
W_SING = 2.0
INVALID_COST = 1e4

HOLDOUT_FIT = 0.7
GRID_N = 128
STRUCT_MUT_P = 0.12
IMMIGRANT_FRAC = 0.08
SHARE_FACTOR = 0.06
STAGNATION_LIMIT = 18
ROBUST_JITTER = 0.006      # gene-space sigma ~ printed-part tolerance
ROBUST_N = 8


def _interleaved_mask(ts, n_windows=6, fit_even=True, total_span=None):
    """Interleaved fit/hold-out windows (E-15). total_span pins the window
    width to the FULL recording, so a truncated evolution time base and the
    full-duration reporting time base produce the SAME split (REVIEW-3: the
    two diverged for recordings longer than max_duration, and 37.7% of
    'held-out' samples had actually been trained on)."""
    T = total_span if total_span is not None else (ts[-1] - ts[0])
    if T <= 0:
        return np.ones_like(ts, dtype=bool)
    w = np.minimum((ts - ts[0]) / (T / n_windows), n_windows - 1e-9).astype(int)
    return (w % 2 == 0) if fit_even else (w % 2 == 1)


def structure_key(genome: Genome):
    """Species signature. For open-graph genomes the joint innovation set IS
    the structure, so it must be part of the key (same key => same gene
    layout => crossover and table batching are structure-safe)."""
    return (genome.topology,
            len(genome.bodies),
            tuple(j.innovation for j in genome.joints),
            tuple(a.kind for a in genome.actuators),
            len(genome.springs))


@dataclass
class GraphResult:
    genome: Genome
    topology: str
    cost: float
    history_best: List[float] = field(default_factory=list)
    history_mean: List[float] = field(default_factory=list)
    history_species: List[int] = field(default_factory=list)
    history_parts: List[float] = field(default_factory=list)
    rmse_fit: float = 0.0
    rmse_holdout: float = 0.0
    robustness: Dict = field(default_factory=dict)
    qp_check: Dict = field(default_factory=dict)
    breakdown: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# structural mutation operators (§6.3)
# ---------------------------------------------------------------------------

def _mutate_structure(genome: Genome, rng, ctx) -> Genome:
    g = Genome.from_dict(genome.to_dict())
    is_graph = g.topology == "mechgraph"
    ops = []
    if not g.springs:
        ops.append("add_spring")
    if len(g.actuators) == 1 and (is_graph or g.topology == "direct_linear"):
        ops.append("add_antagonist")
    for i, sp in enumerate(g.springs):
        if sp.genes.get("k", 0) < 0.03:
            ops.append(("remove_spring", i))
    for i, a in enumerate(g.actuators[1:], start=1):
        if a.genes.get("F", 0) < 0.03:
            ops.append(("remove_actuator", i))
    if is_graph:
        # open structural growth (SPEC-2 §5 / E-10): the Assur dyad is the
        # mobility-preserving primitive, weighted up so structure is explored
        if len(g.bodies) <= 4:
            ops += ["add_dyad", "add_dyad"]
        if g.bodies:
            ops.append("move_attachment")
            ops.append("remove_dyad")
    if not ops:
        return g
    op = ops[rng.integers(len(ops))]
    if op == "add_spring":
        # behaviour-preserving at birth: k gene 0 decodes to k = 0
        g.springs.append(SpringGene(
            id=f"s{len(g.springs)}", kind="torsional", mount={"joint": "output"},
            genes={"k": 0.0, "free": float(rng.random())},
            innovation=next_innovation()))
    elif op == "add_antagonist":
        # zero-capacity secondary actuator; tendons are tension-only
        kind = "tendon" if rng.random() < 0.5 else "linear"
        if is_graph:
            genes = {"ax": float(rng.random()), "ay": float(rng.random()),
                     "bx": float(rng.random()), "by": float(rng.random()),
                     "F": 0.0, "v": float(rng.random()),
                     "drive": float(rng.random())}
        else:
            genes = {"a": float(rng.random()), "alpha": float(rng.random()),
                     "b": float(rng.random()), "beta": float(rng.random()),
                     "F": 0.0, "v": float(rng.random())}
        g.actuators.append(ActuatorGene(
            id=f"a{len(g.actuators)}", kind=kind,
            mount={"body_a": "ground", "body_b": "output"}, genes=genes,
            innovation=next_innovation()))
    elif op == "add_dyad":
        g2 = _op_add_dyad(g, rng, ctx)
        if g2 is None:
            return g                       # dyad did not assemble: no mutation
        g = g2
    elif op == "move_attachment":
        g2 = _op_move_attachment(g, rng, ctx)
        if g2 is None:
            return g
        g = g2
    elif op == "remove_dyad":
        g2 = _op_remove_dyad(g, rng)
        if g2 is None:
            return g
        g = g2
    elif isinstance(op, tuple) and op[0] == "remove_spring":
        g.springs.pop(op[1])
    elif isinstance(op, tuple) and op[0] == "remove_actuator":
        g.actuators.pop(op[1])
    g.meta["struct_mutations"] = genome.meta.get("struct_mutations", 0) + 1
    return g


def _op_add_dyad(g: Genome, rng, ctx):
    """Assur dyad growth: +2 massless bodies +3 joints => mobility unchanged
    by construction and behaviour-preserving at birth (massless, passive).

    The dyad is CONSTRUCTED at the sweep's start angle and its exact pose is
    stored in genome.meta["x0"] as the Newton warm start - without it, the
    solver starts every coupler at the origin and lands on the wrong branch.
    Returns None if the grown linkage does not assemble across the seed ROM."""
    from .opengraph import (COUPLER_SCALE, SEED_ROM, _dec_anchor, _enc_coupler)
    from .topology import BodyGene
    topo = get_topology("mechgraph")
    th_ref = SEED_ROM[0]                     # construct at the sweep start
    s_max = 0.95 * COUPLER_SCALE * ctx.Lp    # encoder range per link
    for _ in range(12):
        agx, agy = float(rng.random()), float(rng.random())
        aox, aoy = float(rng.random()), float(rng.random())
        P1 = _dec_anchor(ctx, "ground", agx, agy)
        lo = _dec_anchor(ctx, "output", aox, aoy)
        phi = np.pi - th_ref
        P2 = np.array([np.cos(phi) * lo[0] - np.sin(phi) * lo[1],
                       np.sin(phi) * lo[0] + np.cos(phi) * lo[1]])
        M = 0.5 * (P1 + P2) + rng.normal(0.0, 0.02, 2)
        l1, l2 = float(np.linalg.norm(M - P1)), float(np.linalg.norm(P2 - M))
        if not (0.02 < l1 < s_max and 0.02 < l2 < s_max):
            continue
        child = Genome.from_dict(g.to_dict())
        kbase = len(child.bodies)
        c1, c2 = f"c{kbase}", f"c{kbase + 1}"
        child.bodies += [BodyGene(id=c1, kind="coupler", genes={"mass": 0.0},
                                  innovation=next_innovation()),
                         BodyGene(id=c2, kind="coupler", genes={"mass": 0.0},
                                  innovation=next_innovation())]
        e00 = _enc_coupler(ctx, (0.0, 0.0))
        e1 = _enc_coupler(ctx, (l1, 0.0))
        e2 = _enc_coupler(ctx, (l2, 0.0))
        child.joints += [
            JointGene(id=f"j{len(child.joints)}", kind="revolute",
                      body_a="ground", body_b=c1,
                      genes={"ax": agx, "ay": agy, "bx": e00[0], "by": e00[1]},
                      innovation=next_innovation()),
            JointGene(id=f"j{len(child.joints) + 1}", kind="revolute",
                      body_a=c1, body_b=c2,
                      genes={"ax": e1[0], "ay": e1[1],
                             "bx": e00[0], "by": e00[1]},
                      innovation=next_innovation()),
            JointGene(id=f"j{len(child.joints) + 2}", kind="revolute",
                      body_a=c2, body_b="output",
                      genes={"ax": e2[0], "ay": e2[1], "bx": aox, "by": aoy},
                      innovation=next_innovation()),
        ]
        a1 = float(np.arctan2(*(M - P1)[::-1]))
        a2 = float(np.arctan2(*(P2 - M)[::-1]))
        x0 = dict(child.meta.get("x0", {}))
        x0[c1] = [float(P1[0]), float(P1[1]), a1]
        x0[c2] = [float(M[0]), float(M[1]), a2]
        child.meta["x0"] = x0
        try:
            tb = topo.build_table(child, ctx,
                                  np.linspace(SEED_ROM[0], SEED_ROM[1], 24))
        except Exception:
            continue
        if bool(tb.valid.all()):
            return child
    return None


def _op_move_attachment(g: Genome, rng, ctx):
    """Re-mount one actuator endpoint onto another body, anchored at the SAME
    world point at mid-ROM (small perturbation, SPEC-2 §5); the child is
    validated by rebuilding its table before it is accepted."""
    from .opengraph import (GROUND_Y_FRAC, OUTPUT_Y_FRAC, SEED_ROM,
                            _enc_coupler)
    topo = get_topology("mechgraph")
    th_mid = 0.5 * (SEED_ROM[0] + SEED_ROM[1])
    movable = [a for a in g.actuators if a.kind != "rotary"]
    if not movable:
        return None
    act = movable[rng.integers(len(movable))]
    side = "a" if rng.random() < 0.5 else "b"
    cur_body = act.mount[f"body_{side}"]
    targets = [b.id for b in g.bodies if b.id != cur_body]
    targets += [b for b in ("ground", "output") if b != cur_body]
    if not targets:
        return None
    tgt = targets[rng.integers(len(targets))]
    poses = topo.solve_body_poses(g, ctx, th_mid)
    from .opengraph import _dec_anchor
    loc_cur = _dec_anchor(ctx, cur_body, act.genes[f"{side}x"],
                          act.genes[f"{side}y"])
    p, phi = poses[cur_body]
    c, s = np.cos(phi), np.sin(phi)
    w = p + np.array([c * loc_cur[0] - s * loc_cur[1],
                      s * loc_cur[0] + c * loc_cur[1]])
    pt, pht = poses[tgt]
    ct, st = np.cos(-pht), np.sin(-pht)
    d = w - pt
    loc_t = np.array([ct * d[0] - st * d[1], st * d[0] + ct * d[1]])
    if tgt == "ground":
        gx = float(np.clip(loc_t[0] / ctx.Lp, 0, 1))
        gy = float(np.clip(loc_t[1] / (GROUND_Y_FRAC * ctx.Lp) / 2 + 0.5, 0, 1))
    elif tgt == "output":
        gx = float(np.clip(loc_t[0] / (0.95 * ctx.Ld), 0, 1))
        gy = float(np.clip(loc_t[1] / (OUTPUT_Y_FRAC * ctx.Ld) / 2 + 0.5, 0, 1))
    else:
        gx, gy = _enc_coupler(ctx, loc_t)
    act.mount[f"body_{side}"] = tgt
    act.genes[f"{side}x"], act.genes[f"{side}y"] = gx, gy
    act.innovation = next_innovation()      # a new force path is new structure
    try:
        tb = topo.build_table(g, ctx, np.linspace(SEED_ROM[0], SEED_ROM[1], 24))
    except Exception:
        return None
    return g if bool(tb.valid.all()) else None


def _op_remove_dyad(g: Genome, rng):
    """Prune a passive massless dyad (simplify): the two bodies must carry no
    actuator mount and near-zero mass; their joints go with them."""
    mounted = {a.mount.get("body_a") for a in g.actuators} | \
              {a.mount.get("body_b") for a in g.actuators}
    cands = [b.id for b in g.bodies
             if b.id not in mounted and float(b.genes.get("mass", 0)) < 0.05]
    pairs = []
    for jd in g.joints:
        if jd.body_a in cands and jd.body_b in cands:
            pairs.append((jd.body_a, jd.body_b))
    if not pairs:
        return None
    ba, bb = pairs[rng.integers(len(pairs))]
    g.bodies = [b for b in g.bodies if b.id not in (ba, bb)]
    g.joints = [j for j in g.joints
                if j.body_a not in (ba, bb) and j.body_b not in (ba, bb)]
    if 3 * len(g.bodies) != 2 * len(g.joints):     # must stay single-DOF
        return None
    return g


def _mutate_params(genome: Genome, rng, sigma: float):
    for path, val in genome.continuous_genes().items():
        if rng.random() < 0.9:
            genome.set_continuous(path, val + rng.normal(0.0, sigma))


def _crossover(g1: Genome, g2: Genome, rng) -> Genome:
    """BLX-alpha on gene paths shared by both parents (same species =>
    identical structure); disjoint genes come from the fitter parent g1."""
    child = Genome.from_dict(g1.to_dict())
    c1, c2 = g1.continuous_genes(), g2.continuous_genes()
    for path in c1:
        if path in c2:
            a, b = c1[path], c2[path]
            lo, hi = min(a, b), max(a, b)
            span = (hi - lo) * 0.3
            child.set_continuous(path, rng.uniform(lo - span, hi + span))
    return child


def _random_genome(rng, ctx, topology: str) -> Genome:
    g = get_topology(topology).random_genome(rng, ctx)
    if rng.random() < 0.3:   # some immigrants arrive with a live spring
        g.springs.append(SpringGene(
            id="s0", kind="torsional", mount={"joint": "output"},
            genes={"k": float(rng.random() * 0.6), "free": float(rng.random())},
            innovation=next_innovation()))
    return g


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _effort_class_cost(kinds, F, v):
    c = 0.0
    for k, Fi, vi in zip(kinds, F, v):
        if k == "rotary":
            c += W_FORCE * Fi / 25.0 + W_SPEED * vi / 8.0
        else:
            c += W_FORCE * Fi / 500.0 + W_SPEED * vi / 0.2
    return c


def _cost_terms(genome, topo, table, ctx, rmse, energy, u_demand_peak, F, v,
                in_rom):
    d = topo.describe(genome)
    sat = float(np.sum(np.maximum(u_demand_peak - F, 0.0) / np.maximum(F, 1e-9)))
    sing = float(1.0 / max(np.max(table.cond[in_rom]), 1.0))
    c = (W_TRACK * rmse
         + _effort_class_cost(table.actuator_kinds, F, v)
         + W_COMPLEX_BODY * d["n_bodies"] + W_COMPLEX_JOINT * d["n_joints"]
         + W_COMPLEX_ACT * d["n_actuators"] + W_COMPLEX_SPRING * d["n_springs"]
         + W_ENERGY * energy
         + W_MASS * float(topo.mass_estimate(genome, ctx))
         + W_SAT * sat
         + W_SING * max(0.0, 0.01 - sing) / 0.01)
    return c if np.isfinite(c) else INVALID_COST


def _evaluate_population(pop, groups, ctx, ts, th_d, thdot_d, th_lo, th_hi,
                         grid, payload=None, err_mask=None):
    """Tables are built per species (structure-correct); simulation batches
    merge every valid candidate of the same shape (n_act, unilateral mask)
    into ONE vectorised integration (§5.3 / R-10)."""
    n = len(pop)
    costs = np.full(n, INVALID_COST, dtype=float)
    tables = [None] * n
    caps = [None] * n
    in_rom = (grid >= th_lo) & (grid <= th_hi)
    for key, idxs in groups.items():
        topo = get_topology(key[0])
        gs = [pop[i] for i in idxs]
        tbs = (topo.build_table_batch(gs, ctx, grid)
               if hasattr(topo, "build_table_batch")
               else [topo.build_table(g, ctx, grid) for g in gs])
        for j, i in enumerate(idxs):
            tables[i] = tbs[j]
            caps[i] = topo.capacities(pop[i], ctx)
    sim_groups = {}
    for i, t in enumerate(tables):
        # universal validity gate (SPEC-2 E-4): every topology, every candidate
        g8 = apply_gate(t, pop[i], get_topology(pop[i].topology), ctx,
                        th_lo, th_hi, caps[i][0], caps[i][1])
        if not g8.ok:
            costs[i] = INVALID_COST + 100.0 * g8.soft   # feasibility gradient (§3.2)
        else:
            sim_groups.setdefault((t.n_act, tuple(t.unilateral.tolist())), []).append(i)
    for _, idxs in sim_groups.items():
        F = np.stack([caps[i][0] for i in idxs])
        v = np.stack([caps[i][1] for i in idxs])
        dg = [getattr(get_topology(pop[i].topology), "drive_genes",
                      lambda g: None)(pop[i]) for i in idxs]
        batch = TableBatch.stack([tables[i] for i in idxs], u_max=F, rate_max=v,
                                 drive_genes=dg)
        sim = simulate_tables(batch, ts, th_d, thdot_d, ctx, payload=payload,
                              err_mask=err_mask)
        for j, i in enumerate(idxs):
            costs[i] = _cost_terms(pop[i], get_topology(pop[i].topology),
                                   tables[i], ctx, float(sim["rmse_deg"][j]),
                                   float(sim["energy_J"][j]),
                                   sim["u_demand_peak"][j], F[j], v[j], in_rom)
    return costs, tables


def simulate_graph_final(genome, ctx, t_obs, theta_obs_deg, payload_t=None,
                         payload_kg=None, allocator="qp", grid_n=256):
    """Full-duration single-design simulation with traces (QP allocation)."""
    topo = get_topology(genome.topology)
    ts, th_d, thdot_d = make_target(t_obs, theta_obs_deg, max_duration=60.0)
    pad = 0.15
    grid = np.linspace(th_d.min() - pad, th_d.max() + pad, grid_n)
    table = topo.build_table(genome, ctx, grid)
    F, v = topo.capacities(genome, ctx)
    dg = [getattr(topo, "drive_genes", lambda g: None)(genome)]
    batch = TableBatch.stack([table], u_max=F[None, :], rate_max=v[None, :],
                             drive_genes=dg)
    payload = None
    if payload_t is not None and payload_kg is not None:
        payload = np.interp(ts, payload_t, payload_kg)
    sim = simulate_tables(batch, ts, th_d, thdot_d, ctx, return_traces=True,
                          payload=payload, allocator=allocator)
    tr = sim["traces"]
    return {
        "t": ts, "theta_sim_deg": np.degrees(tr["theta"][0]),
        "theta_target_deg": np.degrees(tr["theta_target"]),
        "q": tr["q"][0], "u": tr["u"][0],
        "rmse_deg": float(sim["rmse_deg"][0]),
        "u_peak": sim["u_peak"][0], "u_demand_peak": sim["u_demand_peak"][0],
        "rate_peak": sim["rate_peak"][0], "energy_J": float(sim["energy_J"][0]),
        "F_max": F, "v_max": v, "table": table,
        # legacy-shaped aliases for downstream plots (first actuator)
        "L": tr["q"][0][0], "F": tr["u"][0][0],
        "F_peak": float(sim["u_peak"][0][0]),
        "F_demand_peak": float(sim["u_demand_peak"][0][0]),
        "v_peak": float(sim["rate_peak"][0][0]),
    }


def robustness_screen(genome, ctx, ts, th_d, thdot_d, th_lo, th_hi, grid,
                      rng, n=ROBUST_N, jitter=ROBUST_JITTER):
    """§8.3: perturb continuous genes ~ printed tolerances; a design that goes
    invalid under perturbation is rejected outright."""
    topo = get_topology(genome.topology)
    rmses, invalid = [], 0
    for _ in range(n):
        g = Genome.from_dict(genome.to_dict())
        for path, val in g.continuous_genes().items():
            g.set_continuous(path, val + rng.normal(0.0, jitter))
        t = topo.build_table(g, ctx, grid)
        F, v = topo.capacities(g, ctx)
        if not apply_gate(t, g, topo, ctx, th_lo, th_hi, F, v).ok:
            invalid += 1
            continue
        batch = TableBatch.stack([t], u_max=F[None, :], rate_max=v[None, :])
        sim = simulate_tables(batch, ts, th_d, thdot_d, ctx)
        rmses.append(float(sim["rmse_deg"][0]))
    return {"rmse_std_deg": float(np.std(rmses)) if rmses else float("inf"),
            "rmse_mean_deg": float(np.mean(rmses)) if rmses else float("inf"),
            "invalid_fraction": invalid / n}


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def evolve_graph(pose_t, pose_theta_deg, ctx, topology: str = "auto",
                 pop_size: int = 120, generations: int = 90, seed: int = 0,
                 log: Optional[Callable] = None,
                 payload_t=None, payload_kg=None,
                 max_duration: float = 12.0) -> GraphResult:
    rng = np.random.default_rng(seed)
    log = log or (lambda *a: None)

    # hold-out (§8.4, SPEC-2 E-15): INTERLEAVED windows, so the held-out data
    # spans the whole movement (a trailing split lands on the static hold).
    # The target and the transmission grid cover the UNION of both windows.
    ts, th_d, thdot_d = make_target(pose_t, pose_theta_deg,
                                    max_duration=max_duration)
    total_span = float(pose_t[-1] - pose_t[0])   # ONE window basis everywhere
    err_mask = _interleaved_mask(ts, n_windows=6, fit_even=True,
                                 total_span=total_span)
    if payload_t is not None or err_mask.sum() < 100:
        err_mask = np.ones_like(ts, dtype=bool)   # task-augmented / too short
    th_lo, th_hi = float(th_d.min()), float(th_d.max())
    pad = 0.15
    grid = np.linspace(th_lo - pad, th_hi + pad, GRID_N)
    payload = None
    if payload_t is not None and payload_kg is not None:
        payload = np.interp(ts, payload_t, payload_kg)

    # seeding: FROM SCRATCH (SPEC-2 §5). "auto" seeds the open mechgraph
    # representation with the minimal possible mechanism (two bones + one
    # linear actuator, rejection-sampled feasible); every richer structure -
    # dyads, moved attachments, antagonists, springs - must be DISCOVERED by
    # mutation and survive the gate. Archetype classes stay flag-selectable
    # and serve as regression oracles. rotary_direct remains excluded from
    # auto until it models gear ratio + reflected inertia (E-8 / G1).
    if topology == "auto":
        kinds = ["mechgraph"] * pop_size
    else:
        kinds = [topology] * pop_size
    pop = [get_topology(k).random_genome(rng, ctx) for k in kinds]

    best_cost_ever, best_genome = np.inf, None
    species_best: Dict = {}
    species_stag: Dict = {}
    hist_b, hist_m, hist_s, hist_p = [], [], [], []

    for gen in range(generations):
        # --- species-batched evaluation -----------------------------------
        groups: Dict = {}
        for i, g in enumerate(pop):
            groups.setdefault(structure_key(g), []).append(i)
        costs, _ = _evaluate_population(pop, groups, ctx, ts, th_d, thdot_d,
                                        th_lo, th_hi, grid, payload, err_mask)

        order = np.argsort(costs)
        if costs[order[0]] < best_cost_ever:
            best_cost_ever = float(costs[order[0]])
            best_genome = Genome.from_dict(pop[order[0]].to_dict())
        hist_b.append(float(costs[order[0]]))
        hist_m.append(float(np.mean(np.minimum(costs, INVALID_COST))))
        hist_s.append(len(groups))
        hist_p.append(float(np.mean([
            sum(get_topology(g.topology).describe(g)[k] for k in
                ("n_bodies", "n_joints", "n_actuators", "n_springs"))
            for g in pop])))
        if gen % 10 == 0 or gen == generations - 1:
            bk = structure_key(pop[order[0]])
            log(f"  gen {gen:3d}  best {costs[order[0]]:8.2f}  "
                f"species {len(groups):2d}  best-structure {bk[0]}"
                f"/{bk[1]}cpl/{len(bk[3])}act/{bk[4]}spr")

        if gen == generations - 1:
            break

        # --- stagnation + fitness sharing ----------------------------------
        shared = costs.copy()
        alive_keys = []
        for key, idxs in groups.items():
            kbest = float(np.min(costs[idxs]))
            if key not in species_best or kbest < species_best[key] - 1e-6:
                species_best[key], species_stag[key] = kbest, 0
            else:
                species_stag[key] += 1
            for i in idxs:
                shared[i] *= (1.0 + SHARE_FACTOR * (len(idxs) - 1))
            if species_stag[key] < STAGNATION_LIMIT or kbest <= best_cost_ever + 1e-9:
                alive_keys.append(key)

        # offspring quotas (SPEC-2 §6): floor of 1 child for EVERY living
        # species, remainder rank-weighted - tail species never starve and the
        # immigrant count is honoured exactly
        ranked = sorted(alive_keys, key=lambda k: species_best[k])
        n_imm = max(1, int(IMMIGRANT_FRAC * pop_size))
        breed_slots = max(0, pop_size - n_imm - 1 - len(ranked))  # minus elites
        weights = np.array([1.0 / (1 + i) for i in range(len(ranked))])
        weights /= weights.sum() if len(ranked) else 1.0
        quotas = {k: 1 for k in ranked}
        extra = max(0, breed_slots - len(ranked))
        # largest-remainder apportionment (REVIEW-3: per-species int()
        # truncation dumped up to S-1 slots into immigration - 16 immigrants
        # at 18 species against an intended 9)
        raw = weights * extra
        base = np.floor(raw).astype(int)
        rem_slots = extra - int(base.sum())
        rem_order = np.argsort(-(raw - base))
        for j, k in enumerate(ranked):
            quotas[k] += int(base[j]) + (1 if j in set(rem_order[:rem_slots].tolist()) else 0)

        new_pop = [Genome.from_dict(best_genome.to_dict())]   # global elite
        sigma = 0.10 * (1 - gen / generations) + 0.02
        for key in ranked:
            idxs = groups[key]
            sub = sorted(idxs, key=lambda i: shared[i])
            new_pop.append(Genome.from_dict(pop[sub[0]].to_dict()))  # species elite
            for _ in range(quotas[key]):
                if len(new_pop) >= pop_size - n_imm:
                    break
                pa = pop[sub[rng.integers(min(3, len(sub)))]]
                pb = pop[sub[rng.integers(min(5, len(sub)))]]
                child = _crossover(pa, pb, rng)
                _mutate_params(child, rng, sigma)
                if rng.random() < STRUCT_MUT_P:
                    child = _mutate_structure(child, rng, ctx)
                new_pop.append(child)
        while len(new_pop) < pop_size:   # immigrants (§6.5); respect a fixed-topology run
            k = topology if topology != "auto" else "mechgraph"
            new_pop.append(_random_genome(rng, ctx, k))
        pop = new_pop[:pop_size]

    # --- final selection: robustness screen on the top distinct designs ----
    order = np.argsort(costs)
    winner, rob = None, None
    for i in list(order[:4]) + [-1]:
        cand = best_genome if i == -1 else pop[i]
        rob = robustness_screen(cand, ctx, ts, th_d, thdot_d, th_lo, th_hi,
                                grid, np.random.default_rng(seed + 1))
        if rob["invalid_fraction"] == 0.0:
            winner = Genome.from_dict(cand.to_dict())
            break
    if winner is None:   # SPEC-2 E-14: a design that only assembles at nominal
        raise RuntimeError(
            "no evolved design survives manufacturing tolerances "
            f"(+/-{ROBUST_JITTER:.3f} gene jitter, N={ROBUST_N}); the run fails "
            "rather than shipping a razor-edge design - increase generations "
            "or relax the movement")
    log(f"  robustness: rmse std {rob['rmse_std_deg']:.2f} deg, "
        f"invalid under tolerance {100*rob['invalid_fraction']:.0f}%")

    # fit/hold-out + greedy-vs-QP report on the winner (§5.6, §8.4)
    fin_qp = simulate_graph_final(winner, ctx, pose_t, pose_theta_deg,
                                  payload_t, payload_kg, allocator="qp")
    fin_gr = simulate_graph_final(winner, ctx, pose_t, pose_theta_deg,
                                  payload_t, payload_kg, allocator="greedy")
    qp_check = {
        "rmse_qp": fin_qp["rmse_deg"], "rmse_greedy": fin_gr["rmse_deg"],
        "rmse_gap_deg": abs(fin_qp["rmse_deg"] - fin_gr["rmse_deg"]),
        "peak_gap_frac": float(np.max(np.abs(fin_qp["u_peak"] - fin_gr["u_peak"])
                                      / np.maximum(fin_qp["u_peak"], 1e-9))),
    }
    t_full = fin_qp["t"]
    err = np.deg2rad(fin_qp["theta_sim_deg"] - fin_qp["theta_target_deg"])
    fit_w = _interleaved_mask(t_full, n_windows=6, fit_even=True,
                              total_span=total_span)
    rmse_fit = float(np.degrees(np.sqrt(np.mean(err[fit_w] ** 2))))
    rmse_hold = float(np.degrees(np.sqrt(np.mean(err[~fit_w] ** 2)))) \
        if (~fit_w).any() else rmse_fit

    return GraphResult(
        genome=winner, topology=winner.topology, cost=best_cost_ever,
        history_best=hist_b, history_mean=hist_m, history_species=hist_s,
        history_parts=hist_p, rmse_fit=rmse_fit, rmse_holdout=rmse_hold,
        robustness=rob, qp_check=qp_check,
        breakdown={"final_cost": best_cost_ever})
