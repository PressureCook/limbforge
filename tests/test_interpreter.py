"""SPEC-2 R3 acceptance: generic interpreter vs hand-written oracles."""

import time

import numpy as np
import pytest

from prosthetic_pipeline import mechanism as mech
from prosthetic_pipeline import topologies  # noqa: F401
from prosthetic_pipeline.interpreter import GraphSpec, JointDecl, solve_tables
from prosthetic_pipeline.simulate import JointContext
from prosthetic_pipeline.topology import get_topology

CTX = JointContext.from_pose("elbow", 0.30, 0.27, 80.0)
GRID = np.linspace(np.deg2rad(2), np.deg2rad(130), 128)


def _valid_four_bar_seeds(n=6):
    topo = get_topology("four_bar")
    out = []
    for s in range(200):
        g = topo.random_genome(np.random.default_rng(s), CTX)
        t = topo.build_table(g, CTX, GRID)
        if np.all(t.valid):
            out.append(g)
        if len(out) == n:
            break
    assert len(out) >= min(n, 3)
    return out


def test_direct_linear_interpreter_matches_closed_form():
    """§13: generic path vs mechanism.py closed form (the Phase 1 oracle)."""
    topo = get_topology("direct_linear")
    for s in range(8):
        g = topo.random_genome(np.random.default_rng(s), CTX)
        ti = topo.build_table(g, CTX, GRID)
        to = topo.oracle_table(g, CTX, GRID)
        assert np.max(np.abs(ti.q - to.q)) < 1e-12
        assert np.max(np.abs(ti.r - to.r)) < 1e-10
        assert np.max(np.abs(ti.G - to.G)) < 1e-10
        assert np.max(np.abs(ti.I_eff - to.I_eff)) < 1e-10
        assert np.max(np.abs(ti.dI_dtheta)) < 1e-10   # constant inertia, K=0


def test_four_bar_interpreter_matches_oracle_to_1e10():
    """R3 acceptance: q and r agree with the hand-written Newton sweep."""
    topo = get_topology("four_bar")
    genomes = _valid_four_bar_seeds()
    t_or = topo.oracle_table_batch(genomes, CTX, GRID)
    t_in = topo.build_table_batch(genomes, CTX, GRID)
    for a, b in zip(t_in, t_or):
        m = a.valid & b.valid
        assert m.mean() > 0.9
        assert np.max(np.abs(a.q[0][m] - b.q[0][m])) < 1e-10
        assert np.max(np.abs(a.r[0][m] - b.r[0][m])) < 1e-10
        # oracle I/G use np.gradient - agreement is loose there by design
        assert np.max(np.abs(a.I_eff[m] - b.I_eff[m])) < 5e-3
        assert np.max(np.abs(a.G[m] - b.G[m])) < 5e-2


def test_analytic_derivatives_vs_central_difference():
    """§2.3 prototype targets: I_eff' to ~1e-8, r from x' consistent to ~1e-9."""
    topo = get_topology("four_bar")
    g = _valid_four_bar_seeds(1)[0]
    h = 2e-5
    base = topo.build_table(g, CTX, GRID)
    up = topo.build_table(g, CTX, GRID + h)
    dn = topo.build_table(g, CTX, GRID - h)
    m = base.valid & up.valid & dn.valid
    # E-3: analytic dI/dtheta vs central difference of the analytic I_eff
    dI_fd = (up.I_eff - dn.I_eff) / (2 * h)
    assert np.max(np.abs(base.dI_dtheta[m] - dI_fd[m])) < 1e-6
    # r = -dq/dtheta: analytic vs central difference of q
    r_fd = -(up.q[0] - dn.q[0]) / (2 * h)
    assert np.max(np.abs(base.r[0][m] - r_fd[m])) < 1e-8


def test_mobility_and_square_system_checks():
    """E-2/G10: Kutzbach is computed, and non-square graphs are refused."""
    for kind in ("direct_linear", "four_bar", "rotary_direct"):
        topo = get_topology(kind)
        g = topo.random_genome(np.random.default_rng(0), CTX)
        assert topo.mobility(g) == 1
    # an overconstrained graph (extra joint) must be rejected before solving
    topo = get_topology("four_bar")
    g = topo.random_genome(np.random.default_rng(0), CTX)
    spec = topo.graph_spec_batch([g], CTX)
    spec.joints.append(JointDecl(0, np.zeros((1, 2)), 2, np.zeros((1, 2))))
    assert spec.kutzbach() < 1 and not spec.is_square()
    with pytest.raises(ValueError, match="single-DOF"):
        solve_tables(spec, [g], topo, CTX, GRID)


def test_conditioning_is_measured_not_constant():
    """E-5: table.cond must vary for a loop-closing mechanism."""
    topo = get_topology("four_bar")
    g = _valid_four_bar_seeds(1)[0]
    t = topo.build_table(g, CTX, GRID)
    assert np.std(t.cond[t.valid]) > 0.0
    assert np.all(t.cond[t.valid] >= 1.0)


def test_species_batched_budget():
    """§2.5: 120 four-bars on a 128-point grid, batched, in interactive time."""
    topo = get_topology("four_bar")
    rng = np.random.default_rng(1)
    genomes = [topo.random_genome(rng, CTX) for _ in range(120)]
    t0 = time.perf_counter()
    topo.build_table_batch(genomes, CTX, GRID)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"batched precompute took {elapsed:.2f}s for 120 candidates"
