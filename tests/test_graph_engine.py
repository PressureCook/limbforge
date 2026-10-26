"""SPEC Phase 3/4 acceptance: behaviour-preserving mutations, allocation, sizing."""

import numpy as np
import pytest

from prosthetic_pipeline import topologies  # noqa: F401
from prosthetic_pipeline.actuator_design import design_from_requirements
from prosthetic_pipeline.graph_evolve import _mutate_structure
from prosthetic_pipeline.reduced_sim import (TableBatch, allocate_greedy,
                                             allocate_qp, simulate_tables)
from prosthetic_pipeline.simulate import JointContext, make_target
from prosthetic_pipeline.topology import (ActuatorRequirement, Genome,
                                          get_topology)

CTX = JointContext.from_pose("elbow", 0.30, 0.27, 80.0)
GRID = np.linspace(np.deg2rad(-5), np.deg2rad(145), 128)


def _target():
    t = np.linspace(0, 5, 151)
    th = 70 - 55 * np.cos(2 * np.pi * 0.4 * t)
    return make_target(t, th)


def _sim(genome, allocator="greedy"):
    topo = get_topology(genome.topology)
    table = topo.build_table(genome, CTX, GRID)
    F, v = topo.capacities(genome, CTX)
    batch = TableBatch.stack([table], u_max=F[None, :], rate_max=v[None, :])
    ts, th_d, thdot_d = _target()
    return simulate_tables(batch, ts, th_d, thdot_d, CTX, return_traces=True,
                           allocator=allocator)


def test_structural_mutations_behaviour_preserving_at_birth():
    """R-12: add_spring (k=0) and add_antagonist (capacity 0) must leave the
    simulated trajectory identical."""
    topo = get_topology("direct_linear")
    base = topo.random_genome(np.random.default_rng(1), CTX)
    ref = _sim(base)["traces"]["theta"][0]
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(30):
        mut = _mutate_structure(base, rng, CTX)
        key = (len(mut.actuators), len(mut.springs))
        if key in seen or key == (1, 0):
            continue
        seen.add(key)
        got = _sim(mut)["traces"]["theta"][0]
        assert np.max(np.abs(got - ref)) < 1e-9, f"mutation {key} changed behaviour"
    assert len(seen) >= 2, "expected both add_spring and add_antagonist to fire"


def test_qp_allocation_exact_and_close_to_greedy():
    """§5.6: QP satisfies the torque constraint exactly (unsaturated) and the
    greedy population policy stays within tolerance of it."""
    rng = np.random.default_rng(2)
    P, n = 64, 2
    r = rng.uniform(-0.03, 0.03, (P, n))
    u_max = rng.uniform(50, 400, (P, n))
    tau = rng.uniform(-4, 4, P)
    uni = np.array([False, True])
    for alloc in (allocate_qp, allocate_greedy):
        u = alloc(tau, r, u_max, uni)
        assert np.all(u[:, 1] >= -1e-9), "tendon must stay tension-only"
        assert np.all(np.abs(u) <= u_max + 1e-6)
    u_qp = allocate_qp(tau, r, u_max, uni)
    cap = np.abs(r[:, 0]) * u_max[:, 0] + np.where(
        (r[:, 1] * np.sign(tau)) > 0, np.abs(r[:, 1]) * u_max[:, 1], 0)
    feas = np.abs(tau) <= 0.95 * cap
    err = np.abs((r * u_qp).sum(axis=1) - tau)
    assert np.max(err[feas]) < 1e-4, "QP must meet the torque command when feasible"

    # end-to-end: an antagonistic pair tracks equally under both policies
    topo = get_topology("direct_linear")
    g = topo.random_genome(np.random.default_rng(5), CTX)
    rng2 = np.random.default_rng(3)
    for _ in range(20):
        g2 = _mutate_structure(g, rng2, CTX)
        if len(g2.actuators) == 2:
            g = g2
            break
    g.actuators[1].genes["F"] = 0.6   # live antagonist
    sg = _sim(g, "greedy")
    sq = _sim(g, "qp")
    assert abs(sg["rmse_deg"][0] - sq["rmse_deg"][0]) < 0.5
    # per-actuator peaks are allocator-dependent BY DESIGN (the policies split
    # torque differently); the meaningful §5.6 properties are (a) identical
    # tracking - asserted above - and (b) QP is the minimum-effort allocation
    assert float(np.sum(sq["u_peak"] ** 2)) <=         float(np.sum(sg["u_peak"] ** 2)) * 1.10 + 1e-6


def test_four_bar_batch_matches_scalar():
    topo = get_topology("four_bar")
    gs = [topo.random_genome(np.random.default_rng(s), CTX) for s in range(6)]
    batch = topo.build_table_batch(gs, CTX, GRID)
    for g, tb in zip(gs, batch):
        one = topo.build_table(g, CTX, GRID)
        assert np.allclose(one.q, tb.q, atol=1e-10)
        assert np.array_equal(one.valid, tb.valid)


def test_rotary_sizing_path():
    """R-19: rotary requirements produce a gearbox+motor spec section."""
    t = np.linspace(0, 4, 200)
    eff = 8.0 * np.sin(2 * np.pi * 0.5 * t)
    rate = 2.0 * np.cos(2 * np.pi * 0.5 * t)
    req = ActuatorRequirement(
        kind="rotary", peak_effort=8.0, rms_effort=5.0, peak_rate=2.0,
        travel=np.deg2rad(120), effort_trace=eff, rate_trace=rate,
        dt=float(t[1] - t[0]), packaging_envelope=0.06, label="a0")
    spec = design_from_requirements([req], CTX)
    s = spec["actuators"][0]
    assert s["feasible"] and s["kind"] == "rotary"
    assert s["drive"]["ratio"] in (16.0, 25.0, 50.0, 100.0, 160.0)
    assert spec["rollup"]["total_mass_kg"] > 0.1
    assert spec["rollup"]["battery_hours"] > 0


def test_mixed_requirements_rollup():
    t = np.linspace(0, 4, 200)
    lin = ActuatorRequirement(
        kind="linear", peak_effort=300.0, rms_effort=120.0, peak_rate=0.08,
        travel=0.05, effort_trace=200 * np.sin(t), rate_trace=0.05 * np.cos(t),
        dt=float(t[1] - t[0]), packaging_envelope=0.09, q_min=0.09, q_max=0.14)
    rot = ActuatorRequirement(
        kind="rotary", peak_effort=6.0, rms_effort=3.0, peak_rate=3.0,
        travel=2.0, effort_trace=4 * np.sin(t), rate_trace=2 * np.cos(t),
        dt=float(t[1] - t[0]), packaging_envelope=0.05, label="a1")
    spec = design_from_requirements([lin, rot], CTX)
    assert len(spec["actuators"]) == 2
    assert spec["rollup"]["n_actuators"] == 2
    from prosthetic_pipeline.actuator_design import requirements_markdown
    md = requirements_markdown(spec, "direct_linear")
    assert "System rollup" in md and "Actuator a1 (rotary)" in md
