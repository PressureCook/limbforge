"""Physics regression tests for the review findings C1, C2, C3, M1, M2, M5."""

import numpy as np
import pytest

from prosthetic_pipeline.joints import SEGMENT_ANTHROPOMETRY
from prosthetic_pipeline.mechanism import Mechanism
from prosthetic_pipeline.simulate import (G, JointContext, gravity_torque,
                                          simulate_final)

BODY = 80.0


def _ctx(joint):
    a = SEGMENT_ANTHROPOMETRY[joint]
    L = a["len_default"]
    # bone length: for shoulder/hip the bone is shorter than the mass length
    Ld = {"shoulder": 0.30, "hip": 0.42}.get(joint, L)
    return JointContext.from_pose(joint, 0.4, Ld, BODY, L_mass=L)


@pytest.mark.parametrize("joint,neutral_deg", [
    ("elbow", 0.0), ("knee", 0.0), ("hip", 0.0),
    ("shoulder", 180.0), ("ankle", 90.0),
])
def test_gravity_settles_at_anatomical_neutral(joint, neutral_deg):
    """C1: released with no actuator, every joint must fall to its real
    resting pose - the arm hangs (theta -> 180 measured), the foot dangles at 90."""
    ctx = _ctx(joint)
    for start_offset in (-50.0, 40.0):
        theta = np.deg2rad(neutral_deg + start_offset)
        omega = 0.0
        dt = 1e-3
        for _ in range(60000):  # 60 s with strong damping -> steady state
            tau = gravity_torque(theta, ctx) - 20.0 * ctx.damping * omega
            omega += dt * tau / ctx.inertia
            theta += dt * omega
        assert abs(np.degrees(theta) - neutral_deg) < 2.0, \
            f"{joint} settled at {np.degrees(theta):.1f} deg, expected {neutral_deg}"


def test_gravity_peak_at_quarter_turn_from_neutral():
    ctx = _ctx("shoulder")
    # arm horizontal (measured ~90) must carry the peak static moment
    tau_horiz = abs(gravity_torque(np.deg2rad(90.0), ctx))
    tau_hang = abs(gravity_torque(np.deg2rad(178.0), ctx))
    assert tau_horiz > 10 * tau_hang
    assert tau_horiz == pytest.approx(ctx.mass * G * ctx.com, rel=1e-6)


def test_whole_limb_mass_lever(subtests=None):
    """C3/M2: shoulder and hip carry whole-limb mass at whole-limb COM, with
    Winter-consistent inertia (I_prox = I_com + m*com^2 must be satisfiable)."""
    for joint in ("shoulder", "hip"):
        a = SEGMENT_ANTHROPOMETRY[joint]
        ctx = _ctx(joint)
        assert ctx.com == pytest.approx(a["com_frac"] * a["len_default"], rel=1e-9)
        # radius of gyration about the joint must exceed the COM distance
        assert a["k_frac"] > a["com_frac"]
        assert ctx.inertia == pytest.approx(
            ctx.mass * (a["k_frac"] * a["len_default"]) ** 2, rel=1e-9)
    # peak shoulder gravity torque with the real lever (review: ~10.8 N m at 80 kg)
    tau = abs(gravity_torque(np.deg2rad(90.0), _ctx("shoulder")))
    assert 9.0 < tau < 13.0


def _demo_mech():
    return Mechanism(a=0.12, alpha=0.1, b=0.03, beta=-0.2,
                     F_max=400.0, v_max=0.12, L_ret=0.09, stroke=0.05)


def test_power_sign_convention():
    """C2: mechanical power to the limb is -F*v; lifting against gravity must
    show up as positive work over a slow raise."""
    ctx = _ctx("elbow")
    t = np.linspace(0, 3, 91)
    theta = np.linspace(5, 100, 91)  # slow curl up = work against gravity
    final = simulate_final(_demo_mech(), t, theta, ctx)
    v = np.gradient(final["L"], final["t"][1] - final["t"][0])
    P_mech = -final["F"] * v
    settle = final["t"] > 0.5
    assert np.sum(P_mech[settle]) > 0, "lifting must be net positive mechanical work"
    # and the wrong sign would be net negative
    assert np.sum((final["F"] * v)[settle]) < 0


def test_payload_adds_demand_at_distance():
    """M1: the same payload at a longer lever must demand more force."""
    ctx = _ctx("elbow")
    t = np.linspace(0, 4, 121)
    theta = 50 + 40 * np.sin(2 * np.pi * 0.5 * t)
    pay = np.full_like(t, 1.5)
    near = simulate_final(_demo_mech(), t, theta, ctx,
                          payload_t=t, payload_kg=pay, payload_dist=0.10)
    far = simulate_final(_demo_mech(), t, theta, ctx,
                         payload_t=t, payload_kg=pay, payload_dist=0.30)
    assert far["F_demand_peak"] > near["F_demand_peak"] > 0


def test_demand_can_exceed_capacity():
    """M5: F_demand_peak reports how much force was NEEDED, unclipped."""
    ctx = _ctx("elbow")
    weak = Mechanism(a=0.12, alpha=0.1, b=0.03, beta=-0.2,
                     F_max=20.0, v_max=0.3, L_ret=0.09, stroke=0.05)
    t = np.linspace(0, 2, 61)
    theta = 10 + 80 * (t / t[-1])
    final = simulate_final(weak, t, theta, ctx)
    assert final["F_peak"] <= 20.0 + 1e-9
    assert final["F_demand_peak"] > 20.0
