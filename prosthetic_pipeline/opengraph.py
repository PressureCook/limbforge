"""The open mechanism-graph topology (SPEC-2 §5 / E-1, E-10; REVIEW-3 H3).

Unlike the archetype classes in topologies.py, a "mechgraph" genome carries the
mechanism AS DATA - genome.bodies (coupler links), genome.joints (revolute
constraints with local anchors) and genome.actuators - and the GraphSpec is
built directly from that data. Structural mutation can therefore add coupler
dyads (Assur growth: +2 bodies +3 joints keeps mobility exactly 1), move
actuator attachments onto invented bodies, and remove passive structure, so
the search genuinely starts from scratch: seeds are the minimal possible
mechanism (2 bones + 1 actuator) and everything richer is discovered.

Local-anchor decode conventions (all genes in [0,1]):
  ground body:  x = ax * Lp,             y = (2*ay - 1) * 0.40 * Lp
  output body:  x = bx * 0.95 * Ld,      y = (2*by - 1) * 0.40 * Ld
  coupler body: x = (2*ax - 1) * 0.6*Lp, y = (2*ay - 1) * 0.6*Lp
Coupler mass decodes to gene * 0.30 kg (0 at gene 0 => a newly added dyad is
massless and behaviour-preserving at birth); its inertia uses the anchor
spread as the link length.
"""

from typing import Dict, List

import numpy as np

from . import mechanism as mech
from .topologies import (Barrel, Pin, Plate, Rod, SEED_ROM,
                         _decode_primary_capacity, _decode_secondary_capacity,
                         _requirements_linear, _spring_mass, _spring_terms)
from .topology import (ActuatorGene, BodyGene, Genome, JointGene,
                       register_topology)

COUPLER_MASS_MAX = 0.30      # kg at gene 1
COUPLER_SCALE = 0.6          # local anchor range as a fraction of Lp
GROUND_Y_FRAC = 0.40
OUTPUT_Y_FRAC = 0.40


def _dec_ground(ctx, ax, ay):
    return np.array([ax * ctx.Lp, (2 * ay - 1) * GROUND_Y_FRAC * ctx.Lp])


def _dec_output(ctx, ax, ay):
    return np.array([ax * 0.95 * ctx.Ld, (2 * ay - 1) * OUTPUT_Y_FRAC * ctx.Ld])


def _dec_coupler(ctx, ax, ay):
    s = COUPLER_SCALE * ctx.Lp
    return np.array([(2 * ax - 1) * s, (2 * ay - 1) * s])


def _enc_coupler(ctx, xy):
    s = COUPLER_SCALE * ctx.Lp
    return (float(np.clip(xy[0] / s / 2 + 0.5, 0, 1)),
            float(np.clip(xy[1] / s / 2 + 0.5, 0, 1)))


def _dec_anchor(ctx, body_id, ax, ay):
    if body_id == "ground":
        return _dec_ground(ctx, ax, ay)
    if body_id == "output":
        return _dec_output(ctx, ax, ay)
    return _dec_coupler(ctx, ax, ay)


@register_topology("mechgraph")
class MechGraph:
    """Open planar mechanism graph, interpreted directly from the genome."""

    # --- structure -----------------------------------------------------------
    def _body_index(self, genome) -> Dict[str, int]:
        idx = {"ground": 0, "output": 1}
        for k, b in enumerate(genome.bodies):
            idx[b.id] = 2 + k
        return idx

    def mobility(self, genome) -> int:
        """Kutzbach FROM THE GENOME DATA (E-2), not a constant: bodies and
        joints are whatever mutation has made them."""
        n_bodies = 2 + len(genome.bodies)
        n_joints = 1 + len(genome.joints)      # + the anatomical revolute
        return 3 * (n_bodies - 1) - 2 * n_joints

    def graph_spec_batch(self, genomes: List[Genome], ctx):
        from .interpreter import ActuatorDecl, GraphSpec, JointDecl
        P = len(genomes)
        K = len(genomes[0].bodies)
        bidx = self._body_index(genomes[0])
        com = np.zeros((P, K, 2))
        mass = np.zeros((P, K))
        Icom = np.zeros((P, K))
        for p, g in enumerate(genomes):
            # coupler COM ~ mean of its anchors; inertia from anchor spread
            locs = {b.id: [] for b in g.bodies}
            for jd in g.joints:
                for side, keys in (("a", ("ax", "ay")), ("b", ("bx", "by"))):
                    bid = getattr(jd, f"body_{side}")
                    if bid in locs:
                        locs[bid].append(_dec_coupler(ctx, jd.genes[keys[0]],
                                                      jd.genes[keys[1]]))
            for k, b in enumerate(g.bodies):
                pts = locs.get(b.id) or [np.zeros(2)]
                com[p, k] = np.mean(pts, axis=0)
                mass[p, k] = float(b.genes.get("mass", 0.0)) * COUPLER_MASS_MAX
                span = (np.ptp([q[0] for q in pts]) ** 2
                        + np.ptp([q[1] for q in pts]) ** 2) ** 0.5 if len(pts) > 1 else 0.05
                Icom[p, k] = mass[p, k] * max(span, 0.02) ** 2 / 12.0
        joints = []
        for j_i, jd0 in enumerate(genomes[0].joints):
            aa = np.stack([_dec_anchor(ctx, jd0.body_a,
                                       g.joints[j_i].genes["ax"],
                                       g.joints[j_i].genes["ay"]) for g in genomes])
            bb = np.stack([_dec_anchor(ctx, jd0.body_b,
                                       g.joints[j_i].genes["bx"],
                                       g.joints[j_i].genes["by"]) for g in genomes])
            joints.append(JointDecl(bidx[jd0.body_a], aa, bidx[jd0.body_b], bb))
        acts = []
        for a_i, ad0 in enumerate(genomes[0].actuators):
            if ad0.kind == "rotary":
                acts.append(ActuatorDecl(kind="rotary"))
                continue
            pa = np.stack([_dec_anchor(ctx, ad0.mount["body_a"],
                                       g.actuators[a_i].genes["ax"],
                                       g.actuators[a_i].genes["ay"]) for g in genomes])
            pb = np.stack([_dec_anchor(ctx, ad0.mount["body_b"],
                                       g.actuators[a_i].genes["bx"],
                                       g.actuators[a_i].genes["by"]) for g in genomes])
            acts.append(ActuatorDecl(
                kind=ad0.kind, body_a=bidx[ad0.mount["body_a"]], pt_a=pa,
                body_b=bidx[ad0.mount["body_b"]], pt_b=pb))
        return GraphSpec(P=P, n_couplers=K, coupler_com=com,
                         coupler_mass=mass, coupler_Icom=Icom,
                         joints=joints, actuators=acts)

    # --- tables --------------------------------------------------------------
    def build_table_batch(self, genomes, ctx, theta_grid):
        from .interpreter import solve_tables
        spec = self.graph_spec_batch(genomes, ctx)
        if not spec.is_square():
            raise ValueError(
                f"mechgraph genome is not single-DOF (M={spec.kutzbach()})")
        K = spec.n_couplers
        X0 = None
        if K:
            # warm start from the construction pose stored at dyad birth
            # (genome.meta["x0"]), without which Newton starts every coupler
            # at the origin and rarely finds the intended assembly branch
            X0 = np.zeros((len(genomes), 3 * K))
            for p, g in enumerate(genomes):
                x0 = g.meta.get("x0", {})
                for k, b in enumerate(g.bodies):
                    if b.id in x0:
                        X0[p, 3 * k:3 * k + 3] = x0[b.id]
        return solve_tables(spec, genomes, self, ctx, theta_grid, X0=X0)

    def build_table(self, genome, ctx, theta_grid):
        return self.build_table_batch([genome], ctx, theta_grid)[0]

    def solve_body_poses(self, genome, ctx, theta):
        """World (x, y, phi) per body id at one flexion angle (viz + operators)."""
        from .interpreter import _Poses, _assemble
        spec = self.graph_spec_batch([genome], ctx)
        K = spec.n_couplers
        X = np.zeros((1, 3 * K))
        for th in np.linspace(SEED_ROM[0], float(theta), 12):  # warm sweep
            for _ in range(20):
                f, Jx, _ = _assemble(spec, float(th), X)
                if not np.isfinite(f).all() or np.linalg.norm(f) < 1e-12:
                    break
                try:
                    X = X - np.linalg.solve(Jx, f[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    break
        out = {"ground": (np.zeros(2), 0.0),
               "output": (np.zeros(2), np.pi - float(theta))}
        for k, b in enumerate(genome.bodies):
            out[b.id] = (X[0, 3 * k:3 * k + 2].copy(), float(X[0, 3 * k + 2]))
        return out

    # --- capacities / drive --------------------------------------------------
    def capacities(self, genome, ctx):
        F, v = [], []
        for i, a in enumerate(genome.actuators):
            if i == 0:
                Fi, vi = _decode_primary_capacity(a.genes["F"], a.genes["v"])
            else:
                Fi, vi = _decode_secondary_capacity(a.genes["F"], a.genes["v"])
            F.append(Fi)
            v.append(vi)
        return np.array(F), np.array(v)

    def drive_genes(self, genome):
        return [float(a.genes.get("drive", 0.75)) for a in genome.actuators]

    def describe(self, genome) -> dict:
        return {"n_bodies": 2 + len(genome.bodies),
                "n_joints": 1 + len(genome.joints),
                "n_actuators": len(genome.actuators),
                "n_springs": len(genome.springs),
                "kind": "mechgraph"}

    def mass_estimate(self, genome, ctx) -> float:
        F, _ = self.capacities(genome, ctx)
        act = sum(0.12 + 0.25 * Fi / 1500.0 for Fi in F)
        links = sum(float(b.genes.get("mass", 0.0)) * COUPLER_MASS_MAX
                    for b in genome.bodies)
        return act + links + _spring_mass(genome) + 0.05

    def actuator_requirements(self, genome, ctx, table, traces):
        return _requirements_linear(table, self.capacities(genome, ctx), traces)

    # --- downstream (solid model at TRUE evolved dimensions - G9) ------------
    def solid_model(self, genome, ctx, scale):
        parts = []
        mask = table_valid = None
        tb = self.build_table(genome, ctx,
                              np.linspace(SEED_ROM[0], SEED_ROM[1], 48))
        for i, a in enumerate(genome.actuators):
            if a.kind == "rotary":
                continue
            q = tb.q[i][tb.valid] if tb.valid.any() else tb.q[i]
            q_lo, q_hi = float(q.min()), float(q.max())
            stroke_mm = (q_hi - q_lo) * 1000 + 6
            body_mm = stroke_mm + mech.DEAD_LENGTH_PARALLEL * 1000
            parts.append(Barrel(od=22 * scale, bore=8.6 * scale,
                                length=body_mm * scale))
            parts.append(Rod(d=8 * scale, length=(stroke_mm + 40) * scale))
        # one plate per coupler at its true anchor spread; plates for bones
        for b in genome.bodies:
            pts = []
            for jd in genome.joints:
                if jd.body_a == b.id:
                    pts.append(_dec_coupler(ctx, jd.genes["ax"], jd.genes["ay"]))
                if jd.body_b == b.id:
                    pts.append(_dec_coupler(ctx, jd.genes["bx"], jd.genes["by"]))
            if len(pts) >= 2:
                mm = [(1000 * p[0] * scale, 1000 * p[1] * scale) for p in pts]
                parts.append(Plate(outline=mm, bores=mm))
        parts.append(Pin(d=4.0, length=25.0))
        return parts

    def draw(self, ax, genome, ctx, theta):
        poses = self.solve_body_poses(genome, ctx, theta)
        psi = np.pi - theta
        ax.plot([0, ctx.Lp], [0, 0], color="#444444", lw=6,
                solid_capstyle="round")
        ax.plot([0, ctx.Ld * np.cos(psi)], [0, ctx.Ld * np.sin(psi)],
                color="#e05c2a", lw=6, solid_capstyle="round")

        def world(body_id, loc):
            p, phi = poses[body_id]
            c, s = np.cos(phi), np.sin(phi)
            return p + np.array([c * loc[0] - s * loc[1],
                                 s * loc[0] + c * loc[1]])

        for jd in genome.joints:
            pa = world(jd.body_a, _dec_anchor(ctx, jd.body_a,
                                              jd.genes["ax"], jd.genes["ay"]))
            pb = world(jd.body_b, _dec_anchor(ctx, jd.body_b,
                                              jd.genes["bx"], jd.genes["by"]))
            ax.plot(*np.stack([pa, pb], 1), color="#3f7f3f", lw=3)
        for a in genome.actuators:
            if a.kind == "rotary":
                continue
            pa = world(a.mount["body_a"], _dec_anchor(
                ctx, a.mount["body_a"], a.genes["ax"], a.genes["ay"]))
            pb = world(a.mount["body_b"], _dec_anchor(
                ctx, a.mount["body_b"], a.genes["bx"], a.genes["by"]))
            ax.plot(*np.stack([pa, pb], 1), color="#3a6ea5", lw=4)
        ax.set_aspect("equal")

    # --- seeding: the minimal mechanism, rejection-sampled feasible ----------
    def random_genome(self, rng, ctx) -> Genome:
        for _ in range(300):
            g6 = rng.random(6)
            if self._fast_feasible(g6, ctx):
                break
        # map the legacy 6-gene anchor parameterisation into mechgraph mount
        # genes so the from-scratch seed is the simplest possible mechanism
        p = mech.decode(g6, ctx.Lp, ctx.Ld)
        A = np.array(mech.anchor_A(p["a"], p["alpha"]))
        Bl = np.array([p["b"] * np.cos(p["beta"]), p["b"] * np.sin(p["beta"])])
        ax_ = float(np.clip(A[0] / ctx.Lp, 0, 1))
        ay_ = float(np.clip(A[1] / (GROUND_Y_FRAC * ctx.Lp) / 2 + 0.5, 0, 1))
        bx_ = float(np.clip(Bl[0] / (0.95 * ctx.Ld), 0, 1))
        by_ = float(np.clip(Bl[1] / (OUTPUT_Y_FRAC * ctx.Ld) / 2 + 0.5, 0, 1))
        return Genome(topology="mechgraph", actuators=[ActuatorGene(
            id="a0", kind="linear",
            mount={"body_a": "ground", "body_b": "output"},
            genes={"ax": ax_, "ay": ay_, "bx": bx_, "by": by_,
                   "F": float(g6[4]), "v": float(g6[5]),
                   "drive": float(rng.random())},
            innovation=1)])

    @staticmethod
    def _fast_feasible(g6, ctx):
        from .topologies import _direct_linear_fast_feasible
        return _direct_linear_fast_feasible(g6, ctx)
