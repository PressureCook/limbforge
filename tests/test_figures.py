"""SPEC-2 V-8: every core figure factory passes the figure contract."""
import numpy as np
import pytest

from prosthetic_pipeline import figures
from prosthetic_pipeline.mechanism import Mechanism, fit_stroke
from prosthetic_pipeline.simulate import JointContext, simulate_final
from prosthetic_pipeline.demo_data import make_demo_series


@pytest.fixture(scope="module")
def sim():
    pose = make_demo_series(0)
    ctx = JointContext.from_pose("elbow", pose.Lp, pose.Ld, 80.0)
    m = Mechanism(0.12, 0.1, 0.05, -0.2, 600.0, 0.1, 0.0, 0.0)
    L, s = fit_stroke(m.a, m.alpha, m.b, m.beta, 0.1, 2.3)
    m.L_ret, m.stroke = float(L[0]), float(s[0])
    return pose, ctx, m, simulate_final(m, pose.t, pose.theta_deg, ctx)


def _capture(fn, *a, **k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seen = []
    orig = plt.close
    try:
        plt.close = lambda f=None: seen.append(f) or None
        fn(*a, **k)
    finally:
        plt.close = orig
    assert seen, "figure factory closed no figure"
    fig = seen[-1]
    try:
        figures.check(fig)
    finally:
        orig(fig)


def test_mechanism_curves_pass_contract(sim, tmp_path):
    from prosthetic_pipeline.visualize import plot_mechanism_curves
    pose, ctx, m, final = sim
    _capture(plot_mechanism_curves, m, ctx, 0.15, 2.2,
             str(tmp_path / "m.png"))


def test_tracking_passes_contract(sim, tmp_path):
    from prosthetic_pipeline.visualize import plot_tracking
    pose, ctx, m, final = sim
    _capture(plot_tracking, final, m, str(tmp_path / "t.png"))


def test_convergence_passes_contract(tmp_path):
    from prosthetic_pipeline.visualize import plot_convergence
    _capture(plot_convergence, [10, 5, 3], [50, 20, 9],
             str(tmp_path / "c.png"))


def test_contract_rejects_anonymous_reference_line():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="data")
    ax.axhline(0.5, ls="--")
    ax.set_title("t"); ax.set_xlabel("x (s)"); ax.set_ylabel("y (N)")
    with pytest.raises(figures.FigureContractError):
        figures.check(fig)
    plt.close(fig)
