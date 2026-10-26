"""SPEC-2 acceptance: universal gate (R2), fitness corrections (R4 subset)."""

import numpy as np
import pytest

from prosthetic_pipeline import reduced_sim
from prosthetic_pipeline import topologies as topo_mod  # noqa: F401
from prosthetic_pipeline.evolve import _design_penalties
from prosthetic_pipeline.gate import apply_gate
from prosthetic_pipeline.graph_evolve import (_interleaved_mask,
                                              _mutate_structure)
from prosthetic_pipeline.mechanism import decode
from prosthetic_pipeline.reduced_sim import TableBatch, simulate_tables
from prosthetic_pipeline.runner import RunConfig
from prosthetic_pipeline.runner import run as runner_run
from prosthetic_pipeline.simulate import JointContext, make_target
from prosthetic_pipeline.topology import get_topology

CTX = JointContext.from_pose("elbow", 0.30, 0.27, 80.0)
GRID = np.linspace(np.deg2rad(-5), np.deg2rad(145), 160)
TH_LO, TH_HI = np.deg2rad(10), np.deg2rad(130)


def _dl_genome(vec, rng_seed=0):
    topo = get_topology("direct_linear")
    g = topo.random_genome(np.random.default_rng(rng_seed), CTX)
    for k, val in zip(("a", "alpha", "b", "beta", "F", "v"), vec):
        g.actuators[0].genes[k] = float(val)
    return g


def _gate(genome):
    topo = get_topology(genome.topology)
    t = topo.build_table(genome, CTX, GRID)
    F, v = topo.capacities(genome, CTX)
    return apply_gate(t, genome, topo, CTX, TH_LO, TH_HI, F, v)


def test_gate_passes_a_known_good_design():
    """H6: constrained seeding must produce mostly gate-passing designs over
    the seed ROM (replaces the old single hand-picked lucky-seed genome)."""
    from prosthetic_pipeline.topologies import SEED_ROM
    topo = get_topology("direct_linear")
    rng = np.random.default_rng(0)
    grid = np.linspace(SEED_ROM[0], SEED_ROM[1], 64)
    passed = 0
    for _ in range(40):
        g = topo.random_genome(rng, CTX)
        tb = topo.build_table(g, CTX, grid)
        F, v = topo.capacities(g, CTX)
        if apply_gate(tb, g, topo, CTX, grid[0], grid[-1], F, v).ok:
            passed += 1
    assert passed >= 20, f"only {passed}/40 constrained seeds pass the gate"


def test_gate_rejects_tiny_moment_arm_v3():
    """G7-style fixture: nearly-collinear anchors give a vanishing |r|."""
    res = _gate(_dl_genome([0.55, 0.5, 0.0, 0.5, 0.8, 0.6]))  # b at its minimum
    assert not res.ok
    assert any("min|r|" in r for r in res.reasons)
    assert res.soft > 0                      # feasibility gradient, not a cliff


def test_gate_enforces_legacy_constraints_v5_v6_v7():
    """R2 non-regression: every gate-passing direct_linear genome satisfies the
    LEGACY engine's own hard feasibility penalties (cross-implementation)."""
    rng = np.random.default_rng(0)
    topo = get_topology("direct_linear")
    passed = checked = 0
    for _ in range(600):
        g = topo.random_genome(rng, CTX)
        res = _gate(g)
        checked += 1
        if not res.ok:
            continue
        passed += 1
        vec = np.array([[g.actuators[0].genes[k] for k in
                         ("a", "alpha", "b", "beta", "F", "v")]])
        p = decode(vec, CTX.Lp, CTX.Ld)
        pen, _ = _design_penalties(p, CTX, TH_LO, TH_HI)
        assert pen[0] < 25.0, f"gate passed a legacy-infeasible design (pen={pen[0]:.0f})"
    assert passed >= 2, f"gate passed only {passed}/{checked} - too strict to search"
    assert passed < checked, "gate rejected nothing"


def test_gate_rejects_four_bar_geometry_v8_v9():
    """G8 fixture: at least one builder-valid four-bar is rejected for link
    crossings / envelope by the universal gate."""
    topo = get_topology("four_bar")
    geom_rejected = builder_valid = 0
    for seed in range(120):
        g = topo.random_genome(np.random.default_rng(seed), CTX)
        t = topo.build_table(g, CTX, GRID)
        if not t.all_valid_in(TH_LO, TH_HI):
            continue
        builder_valid += 1
        F, v = topo.capacities(g, CTX)
        res = apply_gate(t, g, topo, CTX, TH_LO, TH_HI, F, v)
        if any(("cross" in r) or ("envelope" in r) for r in res.reasons):
            geom_rejected += 1
    assert builder_valid > 0
    assert geom_rejected > 0, "no geometric rejection - V8/V9 not engaging"


def test_secondary_capacity_decodes_to_zero_rate_g5():
    from prosthetic_pipeline.topologies import _decode_secondary_capacity
    F, v = _decode_secondary_capacity(0.0, 0.0)
    assert F == 0.0 and v == 0.0
    F2, _ = _decode_secondary_capacity(0.0, 0.7)
    assert F2 == 0.0


@pytest.mark.parametrize("op_seed", range(20))
def test_structural_mutations_behaviour_preserving_20_seeds(op_seed):
    """E-11: add_spring / add_antagonist leave the trajectory bit-identical."""
    parent = _dl_genome([0.55, 0.8, 0.5, 0.2, 0.85, 0.6])
    rng = np.random.default_rng(op_seed)
    child = _mutate_structure(parent, rng, CTX)
    t_obs = np.linspace(0, 4, 121)
    th_obs = 70 - 55 * np.cos(2 * np.pi * 0.5 * t_obs)
    ts, th_d, thdot_d = make_target(t_obs, th_obs)

    def run(g):
        topo = get_topology(g.topology)
        t = topo.build_table(g, CTX, GRID)
        F, v = topo.capacities(g, CTX)
        b = TableBatch.stack([t], u_max=F[None, :], rate_max=v[None, :])
        return simulate_tables(b, ts, th_d, thdot_d, CTX, return_traces=True)

    tp = run(parent)["traces"]["theta"][0]
    tc = run(child)["traces"]["theta"][0]
    assert np.max(np.abs(tp - tc)) < 1e-9


def test_energy_charges_co_contraction_g6(monkeypatch):
    """§4.1: opposed effort must cost strictly more than the clean allocation
    delivering the same net torque (the net-torque formula reads them equal)."""
    genome = _dl_genome([0.55, 0.8, 0.5, 0.2, 0.85, 0.6])
    g2 = _dl_genome([0.55, 0.8, 0.5, 0.2, 0.85, 0.6])
    from prosthetic_pipeline.topology import ActuatorGene
    g2.actuators.append(ActuatorGene(
        id="a1", kind="linear", mount={"body_a": "ground", "body_b": "output"},
        genes={"a": 0.55, "alpha": 0.4, "b": 0.35, "beta": 0.6, "F": 0.9, "v": 0.9},
        innovation=99))
    topo = get_topology("direct_linear")
    t2 = topo.build_table(g2, CTX, GRID)
    F2, v2 = topo.capacities(g2, CTX)
    b2 = TableBatch.stack([t2], u_max=F2[None, :], rate_max=v2[None, :])
    t_obs = np.linspace(0, 4, 121)
    th_obs = np.linspace(20, 100, 121)          # slow steady flexion
    ts, th_d, thdot_d = make_target(t_obs, th_obs)

    clean = simulate_tables(b2, ts, th_d, thdot_d, CTX)["energy_J"][0]

    real_alloc = reduced_sim.allocate_greedy

    def cocontract2(tau, r, u_max, uni):
        u = real_alloc(tau, r, u_max, uni)
        # inject internal preload: +d on a0, compensating -d*r0/r1 on a1
        d = 120.0
        u2 = u.copy()
        u2[:, 0] += d
        u2[:, 1] -= d * r[:, 0] / np.where(np.abs(r[:, 1]) > 1e-9, r[:, 1], 1e-9)
        return u2

    monkeypatch.setattr(reduced_sim, "allocate_greedy", cocontract2)
    loaded = simulate_tables(b2, ts, th_d, thdot_d, CTX)["energy_J"][0]
    monkeypatch.setattr(reduced_sim, "allocate_greedy", real_alloc)
    assert loaded > clean * 1.5, (loaded, clean)


def test_demand_equals_applied_when_unsaturated():
    genome = _dl_genome([0.55, 0.4, 0.35, 0.6, 0.95, 0.9])   # huge capacity
    topo = get_topology("direct_linear")
    t = topo.build_table(genome, CTX, GRID)
    F, v = topo.capacities(genome, CTX)
    b = TableBatch.stack([t], u_max=F[None, :], rate_max=v[None, :])
    t_obs = np.linspace(0, 4, 121)
    th_obs = 60 - 40 * np.cos(2 * np.pi * 0.3 * t_obs)
    ts, th_d, thdot_d = make_target(t_obs, th_obs)
    sim = simulate_tables(b, ts, th_d, thdot_d, CTX)
    assert sim["u_demand_peak"][0, 0] == pytest.approx(sim["u_peak"][0, 0], rel=1e-9)


def test_spring_decode_grid_independent_e13():
    from prosthetic_pipeline.topology import SpringGene
    g = _dl_genome([0.55, 0.8, 0.5, 0.2, 0.85, 0.6])
    g.springs.append(SpringGene(id="s0", kind="torsional", mount={},
                                genes={"k": 0.5, "free": 0.4}, innovation=7))
    topo = get_topology("direct_linear")
    thetas = np.deg2rad([20.0, 60.0, 100.0])
    S1 = topo.build_table(g, CTX, np.linspace(0.1, 2.4, 64)).S
    S2 = topo.build_table(g, CTX, np.linspace(-0.2, 2.8, 200)).S
    s1 = np.interp(thetas, np.linspace(0.1, 2.4, 64), S1)
    s2 = np.interp(thetas, np.linspace(-0.2, 2.8, 200), S2)
    assert np.allclose(s1, s2, atol=1e-3)


def test_interleaved_holdout_mask_e15():
    ts = np.linspace(0, 12, 1200)
    m = _interleaved_mask(ts, n_windows=6, fit_even=True)
    assert m.any() and (~m).any()
    # both halves span the whole recording, not just its tail
    assert ts[m].max() > 0.8 * ts[-1] and ts[~m].max() > 0.8 * ts[-1]
    assert ts[m].min() < 0.2 * ts[-1] + 2.0 and ts[~m].min() < 4.0


def test_limb_graph_rejected_e12(tmp_path):
    cfg = RunConfig(demo=True, limb="arm", engine="graph",
                    runs_root=str(tmp_path))
    with pytest.raises(ValueError, match="E-12"):
        runner_run(cfg, log=lambda *a: None)
