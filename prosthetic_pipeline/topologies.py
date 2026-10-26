"""Registered topologies (SPEC §5, §7).

- direct_linear : 1-2 linear/tendon actuators spanning the output joint,
                  optional torsional spring (the legacy design is the 1-actuator,
                  0-spring member of this family)
- four_bar      : grounded crank + coupler driving the output link, linear
                  actuator on the crank, optional torsional spring
- rotary_direct : geared rotary drive at the output joint, optional spring

Conventions (shared with mechanism.py): joint at origin, proximal bone along
+x, distal direction u_d(theta) = [cos(pi-theta), sin(pi-theta)]. Gravity acts
along g_hat = [cos(th_n), -sin(th_n)] in this frame, which reproduces the
per-joint neutral convention of simulate.gravity_torque for the output body
and extends it consistently to coupler bodies (SPEC §5.4).

Structural degrees of freedom (springs, second actuator) are initialised
behaviour-preserving at birth (SPEC §6.3): spring k decodes to 0 at gene 0 and
a secondary actuator's capacity decodes to 0 at gene 0.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from . import mechanism as mech
from .simulate import G as GRAV
from .topology import (ActuatorGene, ActuatorRequirement, Genome, JointGene,
                       SpringGene, TransmissionTable, register_topology)

LINK_DENSITY = 0.20      # kg/m, printed/aluminium linkage plate
TRANS_ANGLE_LIM = (np.deg2rad(30), np.deg2rad(150))
COND_MAX = 1e8
TAU_RANGE = (2.0, 200.0)     # N m, rotary capacity decode (log)
OMEGA_RANGE = (0.8, 20.0)    # rad/s


# --- declarative print primitives (SPEC §7.3) --------------------------------
@dataclass
class Plate:
    outline: list            # polyline of (x, y) mm
    bores: list              # (x, y) mm pin holes


@dataclass
class Pin:
    d: float
    length: float


@dataclass
class Barrel:
    od: float
    bore: float
    length: float


@dataclass
class Rod:
    d: float
    length: float


def _u_d(theta):
    return np.stack([np.cos(np.pi - theta), np.sin(np.pi - theta)], axis=-1)


def _g_hat(ctx):
    tn = ctx.theta_neutral
    return np.array([np.cos(tn), -np.sin(tn)])


# ---------------------------------------------------------------------------
# shared decode helpers
# ---------------------------------------------------------------------------

def _decode_primary_capacity(gF, gv):
    flo, fhi = mech.F_MAX_RANGE
    vlo, vhi = mech.V_MAX_RANGE
    return flo * (fhi / flo) ** gF, vlo * (vhi / vlo) ** gv


def _decode_secondary_capacity(gF, gv):
    """Secondary actuators decode FORCE AND RATE to 0 at gene 0, so adding one
    is behaviour-preserving at birth and cannot throttle the joint (SPEC-2 G5)."""
    return float(gF) ** 2 * 800.0, float(gv) ** 2 * 0.35


SPRING_FREE_BOUNDS = (0.0, np.pi)   # rad - fixed physical decode (SPEC-2 E-13)

# generic seeding ROM used for feasibility rejection-sampling (the recording's
# true ROM replaces it at evolve time; seeds just need to be broadly buildable)
SEED_ROM = (np.deg2rad(5.0), np.deg2rad(120.0))


def _segments_cross_np(a0, a1, b0, b1, eps=1e-9):
    d1, d2 = a1 - a0, b1 - b0
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return False
    b = b0 - a0
    t = (b[0] * d2[1] - b[1] * d2[0]) / den
    s = (b[0] * d1[1] - b[1] * d1[0]) / den
    return (eps < t < 1 - eps) and (eps < s < 1 - eps)


def _direct_linear_fast_feasible(gv, ctx):
    """Closed-form gate-lite for seeding (REVIEW-3 H6): moment arm, stroke
    ratio, packaging and actuator-vs-bone crossing, no table build needed."""
    th = np.linspace(*SEED_ROM, 24)
    p = mech.decode(gv, ctx.Lp, ctx.Ld)
    L, dL, r = mech.length_and_derivative(p["a"], p["alpha"], p["b"], p["beta"], th)
    if r.min() < mech.MIN_MOMENT_ARM:
        return False
    q_lo, q_hi = L.min() - mech.STROKE_MARGIN, L.max() + mech.STROKE_MARGIN
    if q_hi / max(q_lo, 1e-9) > mech.STROKE_RATIO_LIMIT:
        return False
    stroke = q_hi - q_lo
    if q_lo < max(mech.MIN_RETRACTED, stroke + mech.DEAD_LENGTH_PARALLEL):
        return False
    A = np.array(mech.anchor_A(p["a"], p["alpha"]))
    prox0, prox1 = np.zeros(2), np.array([ctx.Lp, 0.0])
    for t in th[::3]:
        B = np.array(mech.anchor_B(p["b"], p["beta"], t))
        tip = ctx.Ld * np.array([np.cos(np.pi - t), np.sin(np.pi - t)])
        if _segments_cross_np(A, B, prox0, prox1) or \
                _segments_cross_np(A, B, np.zeros(2), tip):
            return False
    return True


def _gate_seed_batch(topo, ctx, rng, raw_fn, batch=48):
    """Batched rejection-sampling for interpreter-backed topologies (H6):
    build a batch of raw genomes, keep the ones that pass the full gate."""
    from .gate import apply_gate
    grid = np.linspace(*SEED_ROM, 24)
    gs = [raw_fn(rng, ctx) for _ in range(batch)]
    tbs = topo.build_table_batch(gs, ctx, grid)
    passers, best, best_soft = [], gs[0], np.inf
    for g, tb in zip(gs, tbs):
        F, v = topo.capacities(g, ctx)
        gr = apply_gate(tb, g, topo, ctx, grid[0], grid[-1], F, v)
        if gr.ok:
            passers.append(g)
        elif gr.soft < best_soft:
            best, best_soft = g, gr.soft
    return passers, best


def _spring_terms(genome, ctx, theta_grid):
    """Torsional springs at the output joint: S(th) = -sum k (th - th_free).

    The free angle decodes against FIXED physical bounds so the same genome
    means the same spring on every grid (evolution, final run, robustness)."""
    S = np.zeros(len(theta_grid))
    lo, hi = SPRING_FREE_BOUNDS
    for sp in genome.springs:
        k = float(sp.genes.get("k", 0.0)) ** 2 * 4.0 * ctx.mass * GRAV * ctx.com
        free = lo + float(sp.genes.get("free", 0.5)) * (hi - lo)
        S -= k * (theta_grid - free)
    return S


def _spring_mass(genome):
    return sum(0.015 + 0.04 * float(sp.genes.get("k", 0.0)) for sp in genome.springs)


def _requirements_linear(table, capacities, traces):
    F, v = capacities
    out = []
    mask = table.valid if table.valid.any() else np.ones_like(table.valid)
    for i in range(table.n_act):
        q = table.q[i][mask]
        eff = traces["efforts"][i]
        out.append(ActuatorRequirement(
            kind=table.actuator_kinds[i], peak_effort=float(F[i]),
            rms_effort=float(np.sqrt(np.mean(eff ** 2))),
            peak_rate=float(v[i]), travel=float(q.max() - q.min()),
            effort_trace=eff, rate_trace=traces["rates"][i], dt=traces["dt"],
            packaging_envelope=float(q.min()),
            q_min=float(q.min()), q_max=float(q.max()), label=f"a{i}"))
    return out


# ---------------------------------------------------------------------------
# direct_linear family (legacy + tendon/antagonist/spring extensions)
# ---------------------------------------------------------------------------

@register_topology("direct_linear")
class DirectLinear:
    """Linear/tendon actuator(s) spanning the output joint, optional spring."""

    def random_genome(self, rng, ctx) -> Genome:
        """Constrained seeding (REVIEW-3 H6): rejection-sample against the
        closed-form gate-lite so seeds start buildable, not just decodable."""
        g = rng.random(6)
        for _ in range(300):
            cand = rng.random(6)
            if _direct_linear_fast_feasible(cand, ctx):
                g = cand
                break
        return Genome(topology="direct_linear", actuators=[ActuatorGene(
            id="a0", kind="linear",
            mount={"body_a": "ground", "body_b": "output"},
            genes={"a": g[0], "alpha": g[1], "b": g[2], "beta": g[3],
                   "F": g[4], "v": g[5]},
            innovation=1)])

    def _decode_act(self, act, ctx, primary):
        gn = act.genes
        vec = np.array([gn["a"], gn["alpha"], gn["b"], gn["beta"], 0.5, 0.5])
        p = mech.decode(vec, ctx.Lp, ctx.Ld)
        if primary:
            F, v = _decode_primary_capacity(gn["F"], gn["v"])
        else:
            F, v = _decode_secondary_capacity(gn["F"], gn["v"])
        return p, F, v

    def mobility(self, genome) -> int:
        # Kutzbach from structure alone: 2 bodies, the output revolute (E-2);
        # the Jacobian-rank verification happens in the interpreter solve
        return 3 * (2 - 1) - 2 * 1

    def graph_spec_batch(self, genomes, ctx):
        from .interpreter import ActuatorDecl, GraphSpec, GROUND, OUTPUT
        P = len(genomes)
        acts = []
        for ai in range(len(genomes[0].actuators)):
            pa = np.zeros((P, 2))
            pb = np.zeros((P, 2))
            for p_i, g in enumerate(genomes):
                dec, _, _ = self._decode_act(g.actuators[ai], ctx, ai == 0)
                pa[p_i] = mech.anchor_A(dec["a"], dec["alpha"])
                pb[p_i] = (dec["b"] * np.cos(dec["beta"]),
                           dec["b"] * np.sin(dec["beta"]))
            acts.append(ActuatorDecl(kind=genomes[0].actuators[ai].kind,
                                     body_a=GROUND, pt_a=pa,
                                     body_b=OUTPUT, pt_b=pb))
        return GraphSpec(P=P, n_couplers=0,
                         coupler_com=np.zeros((P, 0, 2)),
                         coupler_mass=np.zeros((P, 0)),
                         coupler_Icom=np.zeros((P, 0)), actuators=acts)

    def build_table(self, genome, ctx, theta_grid) -> TransmissionTable:
        return self.build_table_batch([genome], ctx, theta_grid)[0]

    def build_table_batch(self, genomes, ctx, theta_grid):
        from .interpreter import solve_tables
        return solve_tables(self.graph_spec_batch(genomes, ctx), genomes,
                            self, ctx, theta_grid)

    def oracle_table(self, genome, ctx, theta_grid) -> TransmissionTable:
        """Hand-written closed form, retained as the regression oracle (§13)."""
        n = len(theta_grid)
        qs, rs, kinds = [], [], []
        for i, act in enumerate(genome.actuators):
            p, _, _ = self._decode_act(act, ctx, i == 0)
            L, dL, _ = mech.length_and_derivative(
                p["a"], p["alpha"], p["b"], p["beta"], theta_grid)
            qs.append(L)
            rs.append(-dL)
            kinds.append(act.kind)
        tn = ctx.theta_neutral
        return TransmissionTable(
            theta=theta_grid,
            q=np.stack(qs), r=np.stack(rs),
            I_eff=np.full(n, ctx.inertia), dI_dtheta=np.zeros(n),
            G=-ctx.mass * GRAV * ctx.com * np.sin(theta_grid - tn),
            S=_spring_terms(genome, ctx, theta_grid),
            valid=np.ones(n, dtype=bool), cond=np.ones(n),
            payload_dydth=ctx.L_mass * np.sin(theta_grid - tn),
            payload_dpdth2=np.full(n, ctx.L_mass ** 2),
            actuator_kinds=kinds,
            unilateral=np.array([a.kind == "tendon" for a in genome.actuators]))

    def oracle_table_batch(self, genomes, ctx, theta_grid):
        """Closed-form batch build - oracle only (interpreter is the engine)."""
        P = len(genomes)
        n = len(theta_grid)
        n_act = len(genomes[0].actuators)
        tn = ctx.theta_neutral
        G = -ctx.mass * GRAV * ctx.com * np.sin(theta_grid - tn)
        pay_dy = ctx.L_mass * np.sin(theta_grid - tn)
        pay_d2 = np.full(n, ctx.L_mass ** 2)
        qs = np.empty((P, n_act, n))
        rs = np.empty((P, n_act, n))
        for ai in range(n_act):
            gv = np.array([[g.actuators[ai].genes[k] for k in
                            ("a", "alpha", "b", "beta")] + [0.5, 0.5]
                           for g in genomes])
            p = mech.decode(gv, ctx.Lp, ctx.Ld)
            L, dL, _ = mech.length_and_derivative(
                p["a"][:, None], p["alpha"][:, None],
                p["b"][:, None], p["beta"][:, None], theta_grid[None, :])
            qs[:, ai], rs[:, ai] = L, -dL
        kinds = [a.kind for a in genomes[0].actuators]
        uni = np.array([k == "tendon" for k in kinds])
        out = []
        for i, g in enumerate(genomes):
            out.append(TransmissionTable(
                theta=theta_grid, q=qs[i], r=rs[i],
                I_eff=np.full(n, ctx.inertia), dI_dtheta=np.zeros(n),
                G=G, S=_spring_terms(g, ctx, theta_grid),
                valid=np.ones(n, dtype=bool), cond=np.ones(n),
                payload_dydth=pay_dy, payload_dpdth2=pay_d2,
                actuator_kinds=list(kinds), unilateral=uni.copy()))
        return out

    def capacities(self, genome, ctx):
        F, v = [], []
        for i, act in enumerate(genome.actuators):
            _, Fi, vi = self._decode_act(act, ctx, i == 0)
            F.append(Fi)
            v.append(vi)
        return np.array(F), np.array(v)

    def actuator_requirements(self, genome, ctx, table, traces):
        return _requirements_linear(table, self.capacities(genome, ctx), traces)

    def mass_estimate(self, genome, ctx):
        F, _ = self.capacities(genome, ctx)
        return sum(0.09 + 0.00025 * f for f in F if f > 1.0) + _spring_mass(genome)

    def solid_model(self, genome, ctx, scale) -> list:
        parts = []
        for i, act in enumerate(genome.actuators):
            p, F, _ = self._decode_act(act, ctx, i == 0)
            if F <= 1.0:
                continue
            A = np.array(mech.anchor_A(p["a"], p["alpha"])) * 1000 * scale
            B = np.array([p["b"] * np.cos(p["beta"]),
                          p["b"] * np.sin(p["beta"])]) * 1000 * scale
            parts += [Plate(outline=[(0, 0), tuple(A)], bores=[(0, 0), tuple(A)]),
                      Plate(outline=[(0, 0), tuple(B)], bores=[(0, 0), tuple(B)]),
                      Barrel(od=20 * scale, bore=8 * scale, length=60 * scale),
                      Rod(d=6 * scale, length=50 * scale)]
        parts.append(Pin(d=4 * scale, length=18 * scale))
        return parts

    def draw(self, ax, genome, ctx, theta):
        ax.plot([0, ctx.Lp], [0, 0], color="#444444", lw=6, solid_capstyle="round")
        tip = ctx.Ld * _u_d(np.array([theta]))[0]
        ax.plot([0, tip[0]], [0, tip[1]], color="#e05c2a", lw=6, solid_capstyle="round")
        colors = ["#3a6ea5", "#4caf7d"]
        for i, act in enumerate(genome.actuators):
            p, F, _ = self._decode_act(act, ctx, i == 0)
            if F <= 1.0:
                continue
            A = np.array(mech.anchor_A(p["a"], p["alpha"]))
            B = np.array(mech.anchor_B(p["b"], p["beta"], theta))
            ls = "--" if act.kind == "tendon" else "-"
            ax.plot([A[0], B[0]], [A[1], B[1]], color=colors[i % 2], lw=4, ls=ls)
            ax.plot([A[0], B[0]], [A[1], B[1]], "ko", ms=4)
        if genome.springs:
            ax.plot(0, 0, "o", ms=14, mfc="none", mec="#c9a227", mew=2)

    def describe(self, genome) -> dict:
        return {"kind": "direct_linear", "n_bodies": 2, "n_joints": 1,
                "n_actuators": len(genome.actuators),
                "n_springs": len(genome.springs)}


# ---------------------------------------------------------------------------
# four_bar
# ---------------------------------------------------------------------------

@register_topology("four_bar")
class FourBar:
    """GROUND -(rev P0)- CRANK -(rev C)- COUPLER -(rev B)- OUTPUT plus the
    anatomical revolute; linear actuator drives point D on the crank."""

    def random_genome(self, rng, ctx) -> Genome:
        """Constrained seeding (REVIEW-3 H6): random four-bar link lengths
        essentially never assemble AND pass the gate (~1 in 120), so seeds are
        drawn from batched rejection-sampling against the full gate, with a
        best-soft fallback so seeding always terminates."""
        cache = getattr(self, "_seed_cache", None) or []
        if not cache:
            best = None
            for _ in range(6):                     # up to 6 x 48 attempts
                passers, soft_best = _gate_seed_batch(self, ctx, rng,
                                                      self._raw_genome)
                cache.extend(passers)
                best = best or soft_best
                if len(cache) >= 4:
                    break
            if not cache:
                cache = [best]
        self._seed_cache = cache
        return self._seed_cache.pop()

    def _raw_genome(self, rng, ctx) -> Genome:
        g = rng.random(11)
        genes = {"p0_r": g[0], "p0_ang": g[1], "l1": g[2], "l2": g[3],
                 "b_r": g[4], "b_ang": g[5], "a_r": g[6], "a_ang": g[7],
                 "f_d": g[8], "F": g[9], "v": g[10]}
        return Genome(
            topology="four_bar",
            joints=[JointGene(id="j1", kind="revolute", body_a="ground",
                              body_b="crank", genes={}, innovation=2),
                    JointGene(id="j2", kind="revolute", body_a="crank",
                              body_b="coupler", genes={}, innovation=3),
                    JointGene(id="j3", kind="revolute", body_a="coupler",
                              body_b="output", genes={}, innovation=4)],
            actuators=[ActuatorGene(id="a0", kind="linear",
                                    mount={"body_a": "ground", "body_b": "crank"},
                                    genes=genes, innovation=5)])

    def _decode(self, genome, ctx):
        g = genome.actuators[0].genes
        P0 = (0.10 + 0.75 * g["p0_r"]) * ctx.Lp * np.array([
            np.cos((2 * g["p0_ang"] - 1) * 0.7), np.sin((2 * g["p0_ang"] - 1) * 0.7)])
        l1 = (0.05 + 0.45 * g["l1"]) * ctx.Lp
        l2 = (0.05 + 0.55 * g["l2"]) * ctx.Lp
        b_r = (0.05 + 0.55 * g["b_r"]) * ctx.Ld
        b_ang = (2 * g["b_ang"] - 1) * 0.6
        A = (0.06 + 0.64 * g["a_r"]) * ctx.Lp * np.array([
            np.cos((2 * g["a_ang"] - 1) * 0.6), np.sin((2 * g["a_ang"] - 1) * 0.6)])
        f_d = 0.3 + 0.7 * g["f_d"]
        F, v = _decode_primary_capacity(g["F"], g["v"])
        return P0, l1, l2, b_r, b_ang, A, f_d, F, v

    def mobility(self, genome) -> int:
        # 4 bodies (ground, output, crank, coupler), 4 revolutes
        return 3 * (4 - 1) - 2 * 4

    def graph_spec_batch(self, genomes, ctx):
        from .interpreter import (ActuatorDecl, GraphSpec, GROUND, JointDecl,
                                  OUTPUT)
        P = len(genomes)
        dec = [self._decode(g, ctx) for g in genomes]
        P0 = np.array([d[0] for d in dec])
        l1 = np.array([d[1] for d in dec])
        l2 = np.array([d[2] for d in dec])
        b_r = np.array([d[3] for d in dec])
        b_ang = np.array([d[4] for d in dec])
        A = np.array([d[5] for d in dec])
        f_d = np.array([d[6] for d in dec])
        zero = np.zeros((P, 2))
        CRANK, COUPLER = 2, 3
        m1, m2 = LINK_DENSITY * l1, LINK_DENSITY * l2
        return GraphSpec(
            P=P, n_couplers=2,
            coupler_com=np.stack([np.stack([l1 / 2, np.zeros(P)], -1),
                                  np.stack([l2 / 2, np.zeros(P)], -1)], 1),
            coupler_mass=np.stack([m1, m2], 1),
            coupler_Icom=np.stack([m1 * l1 ** 2 / 12, m2 * l2 ** 2 / 12], 1),
            joints=[
                JointDecl(GROUND, P0, CRANK, zero),
                JointDecl(CRANK, np.stack([l1, np.zeros(P)], -1), COUPLER, zero),
                JointDecl(COUPLER, np.stack([l2, np.zeros(P)], -1), OUTPUT,
                          np.stack([b_r * np.cos(b_ang), b_r * np.sin(b_ang)], -1)),
            ],
            actuators=[ActuatorDecl(kind="linear", body_a=GROUND, pt_a=A,
                                    body_b=CRANK,
                                    pt_b=np.stack([f_d * l1, np.zeros(P)], -1))])

    def initial_guess(self, genomes, ctx, theta0):
        """Warm start from the closed-form circle intersection (upper branch)."""
        P = len(genomes)
        X0 = np.zeros((P, 6))
        for i, g in enumerate(genomes):
            P0, A, B, C, D = self._pose(g, ctx, theta0)
            X0[i, 0:2] = P0
            X0[i, 2] = np.arctan2(C[1] - P0[1], C[0] - P0[0])
            X0[i, 3:5] = C
            X0[i, 5] = np.arctan2(B[1] - C[1], B[0] - C[0])
        return X0

    def build_table(self, genome, ctx, theta_grid) -> TransmissionTable:
        return self.build_table_batch([genome], ctx, theta_grid)[0]

    def build_table_batch(self, genomes, ctx, theta_grid):
        from .interpreter import solve_tables
        spec = self.graph_spec_batch(genomes, ctx)
        X0 = self.initial_guess(genomes, ctx, float(theta_grid[0]))
        return solve_tables(spec, genomes, self, ctx, theta_grid, X0=X0)

    def oracle_table_batch(self, genomes, ctx, theta_grid):
        """Hand-written Newton sweep - retained as the regression oracle (§13)."""
        th = theta_grid
        n = len(th)
        P = len(genomes)
        dec = [self._decode(g, ctx) for g in genomes]
        P0 = np.array([d[0] for d in dec])            # (P,2)
        l1 = np.array([d[1] for d in dec])
        l2 = np.array([d[2] for d in dec])
        b_r = np.array([d[3] for d in dec])
        b_ang = np.array([d[4] for d in dec])
        A = np.array([d[5] for d in dec])
        f_d = np.array([d[6] for d in dec])

        phi_out = (np.pi - th)[None, :] + b_ang[:, None]          # (P,N)
        B = b_r[:, None, None] * np.stack([np.cos(phi_out), np.sin(phi_out)], axis=2)
        tgt = B - P0[:, None, :]
        d = np.linalg.norm(tgt, axis=2)                            # (P,N)
        assembles = (np.abs(l1 - l2)[:, None] + 1e-9 < d) & (d < (l1 + l2)[:, None] - 1e-9)

        # closed-form seed at the first grid point, upper branch
        base = np.arctan2(tgt[:, 0, 1], tgt[:, 0, 0])
        cosg = np.clip((l1 ** 2 + d[:, 0] ** 2 - l2 ** 2) / (2 * l1 * np.maximum(d[:, 0], 1e-9)),
                       -1, 1)
        x = base + np.arccos(cosg)                                 # (P,)

        phi = np.zeros((P, n))
        valid = assembles.copy()
        cond = np.ones((P, n))
        dth = float(th[1] - th[0]) if n > 1 else 1.0
        flipped = np.zeros(P, dtype=bool)
        prev_step = np.full(P, np.nan)
        for i in range(n):
            for _ in range(6):                                     # warm-started Newton
                C = P0 + l1[:, None] * np.stack([np.cos(x), np.sin(x)], axis=1)
                CB = C - B[:, i, :]
                res = np.einsum("ij,ij->i", CB, CB) - l2 ** 2
                dC = l1[:, None] * np.stack([-np.sin(x), np.cos(x)], axis=1)
                J = 2 * np.einsum("ij,ij->i", CB, dC)
                step = res / np.where(np.abs(J) > 1e-12, J, 1e-12)
                x = x - np.where(assembles[:, i], step, 0.0)
            C = P0 + l1[:, None] * np.stack([np.cos(x), np.sin(x)], axis=1)
            CB = C - B[:, i, :]
            res = np.einsum("ij,ij->i", CB, CB) - l2 ** 2
            dC = l1[:, None] * np.stack([-np.sin(x), np.cos(x)], axis=1)
            J = 2 * np.einsum("ij,ij->i", CB, dC)
            valid[:, i] &= np.abs(res) < 1e-8
            valid[:, i] &= np.abs(J) > 1e-10
            if i:
                step_i = np.abs(x - phi[:, i - 1])
                # threshold scales with the local sweep rate (SPEC-2 §3.1)
                thr = np.maximum(0.15, 6.0 * np.where(np.isnan(prev_step),
                                                      0.05, prev_step))
                flipped |= assembles[:, i] & (step_i > thr)
                prev_step = step_i
            # a flip invalidates the remainder of the sweep, not one sample
            valid[:, i] &= ~flipped
            cond[:, i] = 1.0 / np.maximum(np.abs(J) / (2 * l1 * (l1 + l2)), 1e-12)
            valid[:, i] &= cond[:, i] < COND_MAX
            phi[:, i] = x

        # kinematics from the implicit-function derivative
        C = P0[:, None, :] + l1[:, None, None] * np.stack([np.cos(phi), np.sin(phi)], axis=2)
        dB = b_r[:, None, None] * np.stack([np.sin(phi_out), -np.cos(phi_out)], axis=2)
        dCdphi = l1[:, None, None] * np.stack([-np.sin(phi), np.cos(phi)], axis=2)
        CB = C - B
        dfdphi = 2 * np.einsum("pij,pij->pi", CB, dCdphi)
        dfdth = -2 * np.einsum("pij,pij->pi", CB, dB)
        dphi = -dfdth / np.where(np.abs(dfdphi) > 1e-10, dfdphi, 1e-10)

        D = P0[:, None, :] + (f_d * l1)[:, None, None] * np.stack(
            [np.cos(phi), np.sin(phi)], axis=2)
        DA = D - A[:, None, :]
        q = np.linalg.norm(DA, axis=2)
        dD = (f_d * l1)[:, None, None] * np.stack([-np.sin(phi), np.cos(phi)], axis=2) \
            * dphi[:, :, None]
        dq = np.einsum("pij,pij->pi", DA, dD) / np.maximum(q, 1e-9)
        r = -dq

        m1 = LINK_DENSITY * l1
        m2 = LINK_DENSITY * l2
        com1 = P0[:, None, :] + 0.5 * l1[:, None, None] * np.stack(
            [np.cos(phi), np.sin(phi)], axis=2)
        com2 = 0.5 * (C + B)
        psi = np.unwrap(np.arctan2(B[:, :, 1] - C[:, :, 1], B[:, :, 0] - C[:, :, 0]), axis=1)
        d_com1 = np.gradient(com1, th, axis=1)
        d_com2 = np.gradient(com2, th, axis=1)
        d_psi = np.gradient(psi, th, axis=1)
        I1 = m1 * l1 ** 2 / 12.0
        I2 = m2 * l2 ** 2 / 12.0
        I_eff = (ctx.inertia
                 + m1[:, None] * np.einsum("pij,pij->pi", d_com1, d_com1)
                 + I1[:, None] * dphi ** 2
                 + m2[:, None] * np.einsum("pij,pij->pi", d_com2, d_com2)
                 + I2[:, None] * d_psi ** 2)
        dI = np.gradient(I_eff, th, axis=1)

        gh = _g_hat(ctx)
        tn = ctx.theta_neutral
        G = (-ctx.mass * GRAV * ctx.com * np.sin(th - tn)[None, :]
             - m1[:, None] * GRAV * (d_com1 @ gh)
             - m2[:, None] * GRAV * (d_com2 @ gh))

        tables = []
        pay_dy = ctx.L_mass * np.sin(th - tn)
        pay_d2 = np.full(n, ctx.L_mass ** 2)
        for p in range(P):
            S = _spring_terms(genomes[p], ctx, th)
            tables.append(TransmissionTable(
                theta=th, q=q[p][None, :], r=r[p][None, :],
                I_eff=I_eff[p], dI_dtheta=dI[p], G=G[p], S=S,
                valid=valid[p], cond=cond[p],
                payload_dydth=pay_dy, payload_dpdth2=pay_d2,
                actuator_kinds=["linear"]))
        return tables

    def capacities(self, genome, ctx):
        *_, F, v = self._decode(genome, ctx)
        return np.array([F]), np.array([v])

    def actuator_requirements(self, genome, ctx, table, traces):
        return _requirements_linear(table, self.capacities(genome, ctx), traces)

    def mass_estimate(self, genome, ctx):
        P0, l1, l2, *_ = self._decode(genome, ctx)
        F, _ = self.capacities(genome, ctx)
        return 0.09 + 0.00025 * F[0] + LINK_DENSITY * (l1 + l2) + _spring_mass(genome)

    def solid_model(self, genome, ctx, scale) -> list:
        P0, l1, l2, b_r, b_ang, A, f_d, F, v = self._decode(genome, ctx)
        s = 1000 * scale
        return [Plate(outline=[(0, 0), tuple(P0 * s)],
                      bores=[(0, 0), tuple(P0 * s), tuple(A * s)]),
                Plate(outline=[(0, 0), (l1 * s, 0)],
                      bores=[(0, 0), (l1 * s, 0), (f_d * l1 * s, 0)]),
                Plate(outline=[(0, 0), (l2 * s, 0)], bores=[(0, 0), (l2 * s, 0)]),
                Plate(outline=[(0, 0), (b_r * s, 0)], bores=[(0, 0), (b_r * s, 0)]),
                Barrel(od=20 * scale, bore=8 * scale, length=60 * scale),
                Rod(d=6 * scale, length=50 * scale),
                Pin(d=4 * scale, length=18 * scale)]

    def pose_points(self, genome, ctx, theta):
        """Gate hook (V8/V9): body points and link segments at one pose."""
        P0, A, B, C, D = self._pose(genome, ctx, theta)
        tip = ctx.Ld * _u_d(np.array([theta]))[0]
        pts = [P0, A, B, C, D]
        segs = [(P0, C, "crank"), (C, B, "coupler"),
                ((0.0, 0.0), (ctx.Lp, 0.0), "bone_prox"),
                ((0.0, 0.0), tuple(tip), "bone_dist")]
        return pts, [(np.asarray(s[0], dtype=float), np.asarray(s[1], dtype=float))
                     for s in segs]

    def _pose(self, genome, ctx, theta):
        P0, l1, l2, b_r, b_ang, A, f_d, *_ = self._decode(genome, ctx)
        B = b_r * np.array([np.cos(np.pi - theta + b_ang), np.sin(np.pi - theta + b_ang)])
        tgt = B - P0
        d = np.linalg.norm(tgt)
        base = np.arctan2(tgt[1], tgt[0])
        cosg = np.clip((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * max(d, 1e-9)), -1, 1)
        phi = base + np.arccos(cosg)
        C = P0 + l1 * np.array([np.cos(phi), np.sin(phi)])
        D = P0 + f_d * l1 * np.array([np.cos(phi), np.sin(phi)])
        return P0, A, B, C, D

    def draw(self, ax, genome, ctx, theta):
        P0, A, B, C, D = self._pose(genome, ctx, theta)
        tip = ctx.Ld * _u_d(np.array([theta]))[0]
        ax.plot([0, ctx.Lp], [0, 0], color="#444444", lw=6, solid_capstyle="round")
        ax.plot([0, tip[0]], [0, tip[1]], color="#e05c2a", lw=6, solid_capstyle="round")
        ax.plot([P0[0], C[0]], [P0[1], C[1]], color="#7a5cc4", lw=4)
        ax.plot([C[0], B[0]], [C[1], B[1]], color="#4caf7d", lw=4)
        ax.plot([A[0], D[0]], [A[1], D[1]], color="#3a6ea5", lw=4)
        for p in (P0, C, B, A, D):
            ax.plot(p[0], p[1], "ko", ms=4)
        if genome.springs:
            ax.plot(0, 0, "o", ms=14, mfc="none", mec="#c9a227", mew=2)

    def describe(self, genome) -> dict:
        return {"kind": "four_bar", "n_bodies": 4, "n_joints": 4,
                "n_actuators": 1, "n_springs": len(genome.springs)}


# ---------------------------------------------------------------------------
# rotary_direct
# ---------------------------------------------------------------------------

@register_topology("rotary_direct")
class RotaryDirect:
    """Geared rotary drive at the output joint (q = -theta, r = +1)."""

    def random_genome(self, rng, ctx) -> Genome:
        g = rng.random(2)
        return Genome(topology="rotary_direct", actuators=[ActuatorGene(
            id="a0", kind="rotary", mount={"joint": "output"},
            genes={"tau": g[0], "omega": g[1]}, innovation=6)])

    def capacities(self, genome, ctx):
        g = genome.actuators[0].genes
        tau = TAU_RANGE[0] * (TAU_RANGE[1] / TAU_RANGE[0]) ** g["tau"]
        om = OMEGA_RANGE[0] * (OMEGA_RANGE[1] / OMEGA_RANGE[0]) ** g["omega"]
        return np.array([tau]), np.array([om])

    def mobility(self, genome) -> int:
        return 3 * (2 - 1) - 2 * 1

    def graph_spec_batch(self, genomes, ctx):
        from .interpreter import ActuatorDecl, GraphSpec
        P = len(genomes)
        return GraphSpec(P=P, n_couplers=0,
                         coupler_com=np.zeros((P, 0, 2)),
                         coupler_mass=np.zeros((P, 0)),
                         coupler_Icom=np.zeros((P, 0)),
                         actuators=[ActuatorDecl(kind="rotary")])

    def build_table(self, genome, ctx, theta_grid) -> TransmissionTable:
        from .interpreter import solve_tables
        return solve_tables(self.graph_spec_batch([genome], ctx), [genome],
                            self, ctx, theta_grid)[0]

    def oracle_table(self, genome, ctx, theta_grid) -> TransmissionTable:
        n = len(theta_grid)
        tn = ctx.theta_neutral
        return TransmissionTable(
            theta=theta_grid,
            q=-theta_grid[None, :], r=np.ones((1, n)),
            I_eff=np.full(n, ctx.inertia), dI_dtheta=np.zeros(n),
            G=-ctx.mass * GRAV * ctx.com * np.sin(theta_grid - tn),
            S=_spring_terms(genome, ctx, theta_grid),
            valid=np.ones(n, dtype=bool), cond=np.ones(n),
            payload_dydth=ctx.L_mass * np.sin(theta_grid - tn),
            payload_dpdth2=np.full(n, ctx.L_mass ** 2),
            actuator_kinds=["rotary"])

    def actuator_requirements(self, genome, ctx, table, traces):
        tau, om = self.capacities(genome, ctx)
        eff = traces["efforts"][0]
        return [ActuatorRequirement(
            kind="rotary", peak_effort=float(tau[0]),
            rms_effort=float(np.sqrt(np.mean(eff ** 2))),
            peak_rate=float(om[0]),
            travel=float(table.theta[-1] - table.theta[0]),
            effort_trace=eff, rate_trace=traces["rates"][0], dt=traces["dt"],
            packaging_envelope=0.06, label="a0")]

    def mass_estimate(self, genome, ctx):
        tau, _ = self.capacities(genome, ctx)
        return 0.12 + 0.012 * tau[0] + _spring_mass(genome)

    def solid_model(self, genome, ctx, scale) -> list:
        tau, _ = self.capacities(genome, ctx)
        d = (30 + 1.2 * tau[0]) * scale
        return [Barrel(od=d, bore=8 * scale, length=0.6 * d),
                Plate(outline=[(0, 0), (60 * scale, 0)], bores=[(0, 0), (60 * scale, 0)]),
                Pin(d=4 * scale, length=18 * scale)]

    def draw(self, ax, genome, ctx, theta):
        tip = ctx.Ld * _u_d(np.array([theta]))[0]
        ax.plot([0, ctx.Lp], [0, 0], color="#444444", lw=6, solid_capstyle="round")
        ax.plot([0, tip[0]], [0, tip[1]], color="#e05c2a", lw=6, solid_capstyle="round")
        ax.plot(0, 0, "o", ms=18, mfc="#3a6ea5", mec="k")
        ax.plot(0, 0, "o", ms=6, mfc="w", mec="k")
        if genome.springs:
            ax.plot(0, 0, "o", ms=24, mfc="none", mec="#c9a227", mew=2)

    def describe(self, genome) -> dict:
        return {"kind": "rotary_direct", "n_bodies": 2, "n_joints": 1,
                "n_actuators": 1, "n_springs": len(genome.springs)}
