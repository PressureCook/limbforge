"""SPEC Phase 1/2 acceptance: reduction correctness, reduced EOM, validity gate."""

import numpy as np
import pytest

from prosthetic_pipeline import mechanism as mech
from prosthetic_pipeline import topologies  # noqa: F401 - triggers registration
from prosthetic_pipeline.reduced_sim import TableBatch, simulate_tables
from prosthetic_pipeline.simulate import (JointContext, make_target,
                                          simulate_population)
from prosthetic_pipeline.topology import (Genome, TransmissionTable,
                                          get_topology, register_topology)

CTX = JointContext.from_pose("elbow", 0.30, 0.27, 80.0)
GRID = np.linspace(np.deg2rad(-5), np.deg2rad(145), 512)


def _legacy_genome(vec):
    topo = get_topology("direct_linear")
    g = topo.random_genome(np.random.default_rng(0), CTX)
    for k, v in zip(("a", "alpha", "b", "beta", "F", "v"), vec):
        g.actuators[0].genes[k] = float(v)
    return g


def test_transmission_ratio_equals_legacy_dL():
    """SPEC §11: r(theta) must equal -dL/dtheta from the closed form to 1e-10."""
    topo = get_topology("direct_linear")
    vec = [0.55, 0.4, 0.35, 0.6, 0.5, 0.5]
    genome = _legacy_genome(vec)
    table = topo.build_table(genome, CTX, GRID)
    p = mech.decode(np.array(vec), CTX.Lp, CTX.Ld)
    L, dL, r = mech.length_and_derivative(p["a"], p["alpha"], p["b"], p["beta"], GRID)
    assert np.max(np.abs(table.q[0] - L)) < 1e-12
    assert np.max(np.abs(table.r[0] - (-dL))) < 1e-10


def test_genome_roundtrip():
    """R-8 (serialisation side): genome -> json dict -> genome -> same table."""
    topo = get_topology("four_bar")
    g1 = topo.random_genome(np.random.default_rng(3), CTX)
    g2 = Genome.from_dict(g1.to_dict())
    assert g1.roundtrip_equal(g2)
    t1 = topo.build_table(g1, CTX, GRID[::8])
    t2 = topo.build_table(g2, CTX, GRID[::8])
    assert np.array_equal(t1.q, t2.q) and np.array_equal(t1.valid, t2.valid)


def test_legacy_through_reduced_engine_matches_closed_form():
    """Phase 1 acceptance: same design, closed-form sim vs table sim, ~identical."""
    vec = [0.55, 0.4, 0.35, 0.6, 0.55, 0.55]
    genome = _legacy_genome(vec)
    topo = get_topology("direct_linear")
    table = topo.build_table(genome, CTX, GRID)
    F, v = topo.capacities(genome, CTX)

    t_obs = np.linspace(0, 6, 181)
    th_obs = 70 - 55 * np.cos(2 * np.pi * 0.4 * t_obs)
    ts, th_d, thdot_d = make_target(t_obs, th_obs)

    p = mech.decode(np.array(vec), CTX.Lp, CTX.Ld)
    params = {k: np.array([p[k2]]) for k, k2 in
              (("a", "a"), ("alpha", "alpha"), ("b", "b"), ("beta", "beta"),
               ("F_max", "F_max"), ("v_max", "v_max"))}
    old = simulate_population(params, ts, th_d, thdot_d, CTX, return_traces=True)

    batch = TableBatch.stack([table], u_max=[[F[0]]], rate_max=[[v[0]]])
    new = simulate_tables(batch, ts, th_d, thdot_d, CTX, return_traces=True)

    th_old = old["traces"]["theta"][0]
    th_new = new["traces"]["theta"][0]
    assert np.max(np.abs(th_old - th_new)) < 5e-4          # rad, interp-limited
    assert abs(old["rmse_deg"][0] - new["rmse_deg"][0]) < 1e-3
    assert new["u_demand_peak"][0, 0] >= new["u_peak"][0, 0]


def test_reduced_eom_includes_dI_term():
    """R-7: integrator must apply 1/2 dI/dth th'^2; removing it must fail
    against an RK4 reference of the exact ODE with theta-varying inertia."""
    n = 512
    th = np.linspace(-2.5, 3.0, n)
    I0 = 0.05
    I = I0 * (1.0 + 0.5 * np.sin(th))
    dI = I0 * 0.5 * np.cos(th)
    G = -2.0 * np.sin(th)

    def make(table_dI):
        t = TransmissionTable(
            theta=th, q=np.zeros((1, n)), r=np.ones((1, n)),
            I_eff=I, dI_dtheta=table_dI, G=G, S=np.zeros(n),
            valid=np.ones(n, dtype=bool), cond=np.ones(n),
            payload_dydth=np.zeros(n), payload_dpdth2=np.zeros(n),
            actuator_kinds=["linear"])
        return TableBatch.stack([t], u_max=[[0.0]], rate_max=[[1e9]])

    th0 = 0.8
    ts = np.arange(0, 1.5, 1 / 4800)
    th_d = np.full_like(ts, th0)          # PD is powerless: u_max = 0
    thdot_d = np.zeros_like(ts)

    res_with = simulate_tables(make(dI), ts, th_d, thdot_d, CTX, return_traces=True)
    res_without = simulate_tables(make(np.zeros(n)), ts, th_d, thdot_d, CTX,
                                  return_traces=True)

    # dense RK4 reference of I th'' + 1/2 I' th'^2 = G - b th'
    def f(y):
        thv, om = y
        Iv = I0 * (1 + 0.5 * np.sin(thv))
        dIv = I0 * 0.5 * np.cos(thv)
        return np.array([om, (-2.0 * np.sin(thv) - CTX.damping * om
                              - 0.5 * dIv * om ** 2) / Iv])

    y = np.array([th0, 0.0])
    dt = 1 / 24000
    ref = []
    for k in range(int(1.5 / dt)):
        if k % 5 == 0:
            ref.append(y[0])
        k1 = f(y); k2 = f(y + dt / 2 * k1)
        k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ref = np.array(ref)[: len(ts)]

    err_with = np.max(np.abs(res_with["traces"]["theta"][0][: len(ref)] - ref))
    err_without = np.max(np.abs(res_without["traces"]["theta"][0][: len(ref)] - ref))
    assert err_with < 2e-3
    assert err_without > 10 * err_with, "removing the dI term must be detectable"


def test_dead_centre_four_bar_rejected():
    """SPEC §5.5 regression: a linkage passing a dead centre inside the ROM
    must be invalid there, gating it out before simulation (R-9)."""
    topo = get_topology("four_bar")
    rng = np.random.default_rng(0)
    rejected = 0
    for seed in range(40):
        g = topo._raw_genome(np.random.default_rng(seed), CTX)   # RAW, not
        # the H6-constrained seeder - this test is about the gate, not seeding
        t = topo.build_table(g, CTX, np.linspace(np.deg2rad(5), np.deg2rad(120), 96))
        if not t.all_valid_in(np.deg2rad(5), np.deg2rad(120)):
            rejected += 1
    assert rejected > 5, "random four-bars should frequently fail the gate"
    # and at least one random four-bar must survive it end to end
    assert rejected < 40, "the gate must not reject everything"


def test_registration_rejects_incomplete_topology():
    with pytest.raises((TypeError, AssertionError, AttributeError)):
        @register_topology("broken")
        class Broken:
            def build_table(self, *a):
                return None
