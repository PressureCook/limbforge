"""Stage 3: evolutionary design of the actuator mechanism.

A genetic algorithm searches over the 6-gene design space (anchor placement on
both segments, actuator force and speed class). Every candidate is judged by
physically simulating it tracking the observed joint trajectory, plus design
penalties: stroke ratio a real linear actuator can deliver, minimum moment arm
across the range of motion, enough torque capacity to hold the limb statically,
and compactness/actuator-size costs.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from . import mechanism as mech
from .simulate import JointContext, gravity_torque, make_target, simulate_population

STATIC_TORQUE_MARGIN = 1.3

# fitness weights
W_TRACK = 2.0        # per degree of RMSE
W_FORCE = 0.6        # per 500 N of actuator force class
W_SPEED = 0.4        # per 0.2 m/s of actuator speed class
W_SIZE = 0.5         # anchor compactness


@dataclass
class EvolutionResult:
    best_genome: np.ndarray
    best_mechanism: "mech.Mechanism"
    best_cost: float
    history_best: List[float] = field(default_factory=list)
    history_mean: List[float] = field(default_factory=list)
    cost_breakdown: dict = field(default_factory=dict)


def _design_penalties(p: dict, ctx: JointContext, th_lo: float, th_hi: float):
    """Vectorized penalty terms independent of the dynamic simulation."""
    L_ret, stroke = mech.fit_stroke(p["a"], p["alpha"], p["b"], p["beta"], th_lo, th_hi)
    ratio = (L_ret + stroke) / np.maximum(L_ret, 1e-6)

    grid = np.linspace(th_lo, th_hi, 60)[None, :]
    _, _, r = mech.length_and_derivative(
        p["a"][:, None], p["alpha"][:, None], p["b"][:, None], p["beta"][:, None], grid)
    r_min = r.min(axis=1)

    # static holdability: F_max * r(theta) must beat gravity torque with margin
    tau_need = STATIC_TORQUE_MARGIN * np.abs(gravity_torque(grid, ctx))
    deficit = np.maximum(tau_need - p["F_max"][:, None] * r, 0.0)
    hold_deficit = (deficit / (tau_need + 1e-9)).max(axis=1)

    relu = lambda x: np.maximum(x, 0.0)
    pen = (
        800.0 * relu(ratio - mech.STROKE_RATIO_LIMIT)
        + 800.0 * relu(mech.MIN_RETRACTED - L_ret) / mech.MIN_RETRACTED
        + 600.0 * relu(mech.MIN_MOMENT_ARM - r_min) / mech.MIN_MOMENT_ARM
        + 200.0 * relu(0.008 - stroke) / 0.008
        + 400.0 * hold_deficit
        # packaging: a real actuator's retracted length must fit stroke + dead length
        + 500.0 * relu((stroke + mech.DEAD_LENGTH_PARALLEL) - L_ret) / mech.DEAD_LENGTH_PARALLEL
        # soft preference for a few mm of real margin, so "fits" is informative
        + 6.0 * relu((stroke + mech.DEAD_LENGTH_PARALLEL + 0.006) - L_ret) / 0.006
    )
    return pen, {"L_ret": L_ret, "stroke": stroke, "ratio": ratio, "r_min": r_min,
                 "hold_deficit": hold_deficit}


def _evaluate(genes: np.ndarray, ctx: JointContext, ts, th_d, thdot_d,
              th_lo: float, th_hi: float, payload=None):
    p = mech.decode(genes, ctx.Lp, ctx.Ld)
    pen, aux = _design_penalties(p, ctx, th_lo, th_hi)
    sim = simulate_population(p, ts, th_d, thdot_d, ctx, payload=payload)
    size = (p["a"] + p["b"]) / (0.35 * (ctx.Lp + ctx.Ld))
    cost = (
        W_TRACK * sim["rmse_deg"]
        + W_FORCE * p["F_max"] / 500.0
        + W_SPEED * p["v_max"] / 0.2
        + W_SIZE * size
        + pen
    )
    cost = np.where(np.isfinite(cost), cost, 1e9)
    return cost, sim, pen


def evolve(pose_t: np.ndarray, pose_theta_deg: np.ndarray, ctx: JointContext,
           pop_size: int = 160, generations: int = 120, seed: int = 0,
           log: Optional[Callable[[str], None]] = None,
           payload_t: Optional[np.ndarray] = None,
           payload_kg: Optional[np.ndarray] = None,
           max_duration: float = 12.0) -> EvolutionResult:
    rng = np.random.default_rng(seed)
    ts, th_d, thdot_d = make_target(pose_t, pose_theta_deg, max_duration=max_duration)
    th_lo, th_hi = float(th_d.min()), float(th_d.max())
    payload = None
    if payload_t is not None and payload_kg is not None:
        payload = np.interp(ts, payload_t, payload_kg)

    genes = rng.random((pop_size, mech.N_GENES))
    history_best, history_mean = [], []
    n_elite = 2

    for gen in range(generations):
        cost, _, _ = _evaluate(genes, ctx, ts, th_d, thdot_d, th_lo, th_hi, payload)
        order = np.argsort(cost)
        genes, cost = genes[order], cost[order]
        history_best.append(float(cost[0]))
        history_mean.append(float(np.mean(cost)))
        if log and (gen % 10 == 0 or gen == generations - 1):
            log(f"  gen {gen:3d}  best cost {cost[0]:8.2f}   population mean {np.mean(cost):10.1f}")
        if gen == generations - 1:
            break

        # tournament selection (k=3) over the sorted population
        def pick(n):
            cand = rng.integers(0, pop_size, size=(n, 3))
            return genes[cand.min(axis=1)]  # sorted -> lower index = fitter

        n_child = pop_size - n_elite
        pa, pb = pick(n_child), pick(n_child)
        # BLX-alpha blend crossover
        alpha_blx = 0.3
        lo = np.minimum(pa, pb) - alpha_blx * np.abs(pa - pb)
        hi = np.maximum(pa, pb) + alpha_blx * np.abs(pa - pb)
        children = lo + rng.random(pa.shape) * (hi - lo)
        # gaussian mutation with decaying sigma
        sigma = 0.12 * (1.0 - gen / generations) + 0.02
        mutate = rng.random(children.shape) < 0.9
        children = children + mutate * rng.normal(0.0, sigma, children.shape)
        genes = np.vstack([genes[:n_elite], np.clip(children, 0.0, 1.0)])

    best = genes[0]
    best_mech = mech.make_mechanism(best, ctx.Lp, ctx.Ld, th_lo, th_hi)
    # cost breakdown for the report
    c, sim, pen = _evaluate(best[None, :], ctx, ts, th_d, thdot_d, th_lo, th_hi, payload)
    breakdown = {
        "total": float(c[0]),
        "tracking_rmse_deg": float(sim["rmse_deg"][0]),
        "penalties": float(pen[0]),
        "F_peak_in_eval_N": float(sim["F_peak"][0]),
    }
    return EvolutionResult(
        best_genome=best, best_mechanism=best_mech, best_cost=float(c[0]),
        history_best=history_best, history_mean=history_mean,
        cost_breakdown=breakdown)
