"""Full-body coupled design: the couplings must be real physics, not labels.

1. Carrying a designed distal joint's hardware must RAISE the proximal
   joint's static load (and its evolved force class, checked end-to-end).
2. Recorded distal flexion must LOWER the proximal load when the limb folds
   toward the joint (elbow curls pull the arm COM shoulder-ward).
3. A straight (non-moving) distal joint must change nothing.
"""
import numpy as np
import pytest

from prosthetic_pipeline.body import (coupled_context, hardware_mass_kg,
                                      trajectory_coupling)
from prosthetic_pipeline.demo_data import make_limb_demo_series
from prosthetic_pipeline.simulate import (G, JointContext, gravity_torque,
                                          simulate_final)
from prosthetic_pipeline.mechanism import Mechanism, fit_stroke


@pytest.fixture(scope="module")
def series():
    return make_limb_demo_series("body", seed=1)


def test_hardware_rollup_raises_proximal_static_load(series):
    fake_elbow = {"spec": {"variants": {"ballscrew": {"mass_est_kg": 0.30}},
                           "recommended": "ballscrew"}}
    bare, _ = coupled_context("shoulder", series, {}, 80.0)
    loaded, carried = coupled_context("shoulder", series,
                                      {"elbow": fake_elbow}, 80.0)
    assert carried and carried[0]["joint"] == "elbow"
    th = np.deg2rad(90.0)                       # arm horizontal: peak gravity
    tq_bare = abs(float(gravity_torque(th, bare)))
    tq_loaded = abs(float(gravity_torque(th, loaded)))
    expected_extra = 0.30 * G * series["shoulder"].Ld
    assert tq_loaded > tq_bare
    assert abs((tq_loaded - tq_bare) - expected_extra) < 0.05 * expected_extra
    assert loaded.inertia > bare.inertia


def test_trajectory_coupling_signs_and_magnitude(series):
    t, payload, info = trajectory_coupling("shoulder", series, 80.0)
    assert t is not None and info["from"] == "elbow"
    # elbow flexion can only pull the COM toward the shoulder: payload <= 0
    assert payload.max() <= 1e-9
    assert payload.min() < -1e-3            # and the demo curls deeply
    # magnitude bounded by moving the whole forearm+hand mass to the shoulder
    assert abs(payload.min()) <= 0.022 * 80.0


def test_straight_distal_changes_nothing(series):
    s = {k: v for k, v in series.items()}
    import copy
    st = copy.deepcopy(series["elbow"])
    st.theta_deg = np.zeros_like(st.theta_deg)      # elbow held straight
    s["elbow"] = st
    t, payload, _ = trajectory_coupling("shoulder", s, 80.0)
    assert np.abs(payload).max() < 1e-9


def test_coupling_reduces_shoulder_demand_when_elbow_curls(series):
    sh = series["shoulder"]
    ctx = JointContext.from_pose("shoulder", sh.Lp, sh.Ld, 80.0,
                                 L_mass=sh.L_mass)
    m = Mechanism(0.14, 0.15, 0.06, -0.1, 2000.0, 0.3, 0.0, 0.0)
    lo, hi = np.deg2rad(sh.rom_deg[0]), np.deg2rad(sh.rom_deg[1])
    L, s_ = fit_stroke(m.a, m.alpha, m.b, m.beta, lo, hi)
    m.L_ret, m.stroke = float(L[0]), float(s_[0])
    t, payload, _ = trajectory_coupling("shoulder", series, 80.0)
    static = simulate_final(m, sh.t, sh.theta_deg, ctx)
    coupled = simulate_final(m, sh.t, sh.theta_deg, ctx,
                             payload_t=t, payload_kg=payload)
    # folding the elbow can only reduce the shoulder's force demand. The
    # instantaneous peak can clip at capacity near a small-moment-arm pose,
    # so compare the unclipped-mean effort over the movement.
    assert float(np.mean(np.abs(coupled["F"]))) < \
        float(np.mean(np.abs(static["F"])))
